#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

import phydrax as phx
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementFieldTransfer,
    FiniteElementPlan,
    lagrange_element,
    prepare_nested_field_transfer,
    SourceRealizationFieldSemantics,
)
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)
from phydrax.discretization.iga._adaptive import (
    DWREstimate,
    DWRPollutionCategory,
    QoICertificate,
)
from phydrax.discretization.iga._identity import OverlayCellId
from phydrax.equations import MaterialSiteId, MaterialState, MaterialTransaction
from phydrax.geometry.design._qualification import (
    DerivativeTier,
    DesignQualificationEvidence,
)
from phydrax.lifecycle import Composition, CompositionEntry, CompositionRebind
from phydrax.meshing import CellMeshingResult, MeshAdaptationResult
from phydrax.meshing._adaptation import MeshAdaptationPolicy, MeshAdaptationRoute
from phydrax.meshing._decision import (
    AdaptationAction,
    DecisionBudget,
    FixedEpochDerivativeEvidence,
    MeasuredAdaptationCost,
    PhysicalErrorEvidence,
    RouteFeasibility,
    SolverAwareCandidate,
    SolverAwareDecision,
)
from phydrax.meshing._measurements import NativeExecutionRecord
from phydrax.meshing._proposals import (
    LearnedMeshProposer,
    mesh_proposal_scope,
    MeshCoordinateProposal,
    MeshMarkingProposal,
    MeshMetricProposal,
    MeshProposalFeatures,
    MeshProposalSafetyPolicy,
    prepare_mesh_proposal,
    project_mesh_proposal,
)
from phydrax.solver import FiniteElementAcceptedState, FiniteElementTopologyTransaction
from tests._ported_models import full_port, in_order, PortedAffine
from tests._support.cell_meshes import bisect, kuhn_cube, kuhn_cube_arrays
from tests._support.forced_devices import run_on_forced_devices
from tests._support.geometry_transfer_cases import (
    curved_shear_source,
    shear_term,
    transfer_triangles,
)
from tests._support.solver_campaigns_common import (
    error_and_integral,
    exact,
    HPHistoryTransfer,
    HPPhysicalCertifier,
    MetricPhysicalCertifier,
    MetricPhysicalReanalysis,
    pack,
    peak_rss_bytes,
    physical,
    SolvedFieldRefresh,
    source,
    unpack,
)
from tests._support.solver_hp_campaign import (
    HPSource,
    HPTarget,
    prepare_hp_source,
    prepare_hp_target,
    resume_hp,
    run,
)
from tests._support.solver_metric_campaign import run_metric


# Fixed empirical allowances from test_solver_aware_decisions.py
# at a6827a9e5; no meshing acceptance policy is relaxed.
# Projected learned coordinate displacement against the active bound 0.05.
_LEARNED_DISPLACEMENT_ATOL = 1e-12
# Family-preserving marked adaptation prolongation against the independent coarse-field
# image.
_FAMILY_TRANSFER_ATOL = 2e-12
# Physical-field projection and transferred FE solution against the independent
# stiffness-plus-mass reference.
_PIPELINE_VALUE_ATOL = 3e-12
# Remap-plus-PDE-refresh coefficients against the newly reanalyzed accepted field.
_REANALYZED_FIELD_ATOL = 2e-11
# Solution and primal/dual pairing of the scale-1.2 FE stiffness-plus-mass
# continuation.
_PIPELINE_VALUE_RTOL = 3e-11
# JVP/VJP against independent dense sensitivity and adjoint solves at scale 1.2.
_PIPELINE_DERIVATIVE_RTOL = 3e-9
# JVP/VJP against independent dense sensitivity and adjoint solves at scale 1.2.
_PIPELINE_DERIVATIVE_ATOL = 3e-11
# Centered finite difference with step 1e-5 and tangent direction 0.37.
_PIPELINE_DIFFERENCE_RTOL = 3e-7
# Centered finite difference with step 1e-5 and tangent direction 0.37.
_PIPELINE_DIFFERENCE_ATOL = 3e-9


# Fixed empirical residual bounds from test_solver_aware_decisions.py
# at a6827a9e5; each comparison retains its strictness.
# HP/metric campaign field/integral and JVP/VJP/duality errors against independent
# retained-field references.
_CAMPAIGN_TRANSFER_RESIDUAL = 1e-10
# Metric campaign JVP/VJP versus finite differences on the same fixed transfer route.
_METRIC_TRANSFER_DIFFERENCE_RESIDUAL = 1e-7


# Fixed scalar-reference allowances from test_solver_aware_decisions.py
# at a6827a9e5; values and relative/absolute roles are unchanged.
# Campaign solution/history integrals 1/3, 1/6, 2/3 and L2 errors sqrt(1/180),
# sqrt(1/5) versus closed moments.
_CAMPAIGN_CLOSED_MOMENT_ATOL = 1e-10


def _error(
    revision: str,
    *,
    field: float = 0.01,
    geometry: float = 0.01,
    estimator: str = "trial-estimator",
    objective: str = "energy-error",
) -> PhysicalErrorEvidence:
    return PhysicalErrorEvidence(
        revision,
        objective,
        estimator,
        field_error=field,
        geometry_error=geometry,
        algebraic_error=0.005,
        transfer_error=0.005,
    )


def _budget(*, wall: float = 10.0) -> DecisionBudget:
    return DecisionBudget(
        tolerance=0.1,
        maximum_wall_seconds=wall,
        maximum_memory_bytes=4096,
        maximum_dofs=100,
        maximum_condition=100.0,
    )


def _route(
    candidate: str,
    *,
    source: str = "accepted",
    compiled: str | None = "compiled-trial",
    condition: float = 5.0,
    dofs: int = 20,
    required: tuple[str, ...] = ("field/u", "history/plastic", "materials"),
    dispositions: tuple[tuple[str, str], ...] = (
        ("field/u", "field-transport"),
        ("history/plastic", "history-transport"),
        ("materials", "material-transport"),
    ),
) -> RouteFeasibility:
    return RouteFeasibility(
        source,
        f"target/{candidate}",
        f"prepared/{candidate}",
        candidate,
        cell_families=("triangle",),
        geometry_layout_id="quadratic-coordinate-map",
        field_layouts=(("u", "prepared-field-space"),),
        compiled_layout_id=compiled,
        geometry_certificate_id="source-fidelity-certificate",
        topology_certificate_id="native-topology-certificate",
        required_state_ids=required,
        state_dispositions=dispositions,
        dofs=dofs,
        condition_estimate=condition,
    )


def _candidate(
    identity: str,
    action: AdaptationAction,
    *,
    feasibility: RouteFeasibility | None = None,
    field: float = 0.01,
    geometry: float = 0.01,
    seconds: float = 0.1,
    memory: int = 1024,
) -> SolverAwareCandidate:
    route = _route(identity) if feasibility is None else feasibility
    cost = MeasuredAdaptationCost(
        f"observed/{identity}",
        identity,
        preparation_seconds=seconds,
        compilation_seconds=0.01,
        solve_seconds=0.01,
        transfer_seconds=0.01,
        reanalysis_seconds=0.01,
        decision_seconds=0.01,
        peak_memory_bytes=memory,
    )
    return SolverAwareCandidate(
        action,
        identity,
        route,
        _error(route.target_revision_id, field=field, geometry=geometry),
        cost,
    )


@pytest.mark.parametrize(
    ("field", "geometry", "field_target", "geometry_target", "winner"),
    [(0.03, 0.6, 0.4, 0.01, "geometry"), (0.6, 0.03, 0.01, 0.4, "field")],
    ids=["geometry-dominated", "field-dominated"],
)
def test_decision_reaches_physical_target_without_conflating_field_and_geometry_order(
    field: float,
    geometry: float,
    field_target: float,
    geometry_target: float,
    winner: str,
) -> None:
    baseline = _error("accepted", field=field, geometry=geometry)
    field_route = _candidate(
        "field", AdaptationAction.P, field=0.01, geometry=field_target
    )
    geometry_route = _candidate(
        "geometry",
        AdaptationAction.GEOMETRY_ORDER,
        field=geometry_target,
        geometry=0.01,
    )
    decision = SolverAwareDecision(baseline, _budget(), (geometry_route, field_route))
    selected = decision.selected
    assert selected is not None and selected.candidate_id == winner
    rejected = "field" if winner == "geometry" else "geometry"
    assert "physical-target-unmet" in dict(decision.dispositions)[rejected]
    assert selected.error.bound <= decision.budget.tolerance


@pytest.mark.parametrize(
    ("failure", "issue"),
    [
        ("target", "physical-target-unmet"),
        ("compiler", "compiled-layout-unavailable"),
        ("condition", "conditioning-budget"),
        ("memory", "memory-budget"),
        ("dofs", "dof-budget"),
        ("stale", "stale-source"),
        ("history", "state-disposition-incomplete"),
        ("materials", "state-disposition-incomplete"),
    ],
    ids=[
        "physical-target",
        "compiler-layout",
        "conditioning",
        "peak-memory",
        "dofs",
        "stale-source",
        "missing-history",
        "missing-materials",
    ],
)
def test_cheap_inadmissible_candidate_cannot_win(failure: str, issue: str) -> None:
    dispositions = tuple(
        pair
        for pair in _route("cheap").state_dispositions
        if pair[0]
        != {"history": "history/plastic", "materials": "materials"}.get(failure)
    )
    cheap = _candidate(
        "cheap",
        AdaptationAction.H,
        feasibility=_route(
            "cheap",
            compiled=None if failure == "compiler" else "compiled-trial",
            source="stale-accepted" if failure == "stale" else "accepted",
            condition=101.0 if failure == "condition" else 5.0,
            dofs=101 if failure == "dofs" else 20,
            dispositions=dispositions,
        ),
        geometry=0.2 if failure == "target" else 0.01,
        memory=4097 if failure == "memory" else 1024,
    )
    feasible = _candidate("feasible", AdaptationAction.P, seconds=0.5)
    decision = SolverAwareDecision(
        _error("accepted", field=0.8), _budget(), (cheap, feasible)
    )
    assert decision.selected is feasible
    assert issue in dict(decision.dispositions)["cheap"]
    with pytest.raises(ValueError, match="requested source, action, and target"):
        decision.require_selected("accepted", "cheap", "target/cheap")


def test_failed_trials_count_against_campaign_budget() -> None:
    rejected = _candidate("failed", AdaptationAction.H, geometry=0.3, seconds=0.6)
    feasible = _candidate("feasible", AdaptationAction.P, seconds=0.6)
    decision = SolverAwareDecision(
        _error("accepted", field=0.8),
        _budget(wall=1.0),
        (rejected, feasible),
    )
    assert decision.selected is None
    assert "campaign-wall-budget" in dict(decision.dispositions)["feasible"]
    with pytest.raises(ValueError, match="got no selected route"):
        decision.require_selected("accepted", "feasible", "target/feasible")


def test_failed_and_coincident_trials_remain_in_observed_campaign_budget() -> None:
    candidate = _candidate("feasible", AdaptationAction.P, seconds=0.4)
    decision = SolverAwareDecision(
        _error("accepted", field=0.3),
        _budget(wall=1.0),
        (candidate,),
        observed_campaign_seconds=1.1,
    )
    assert decision.selected is None
    assert dict(decision.dispositions)["feasible"] == ("campaign-wall-budget",)


def test_observed_campaign_cannot_erase_priced_route_work() -> None:
    candidate = _candidate("feasible", AdaptationAction.P, seconds=0.4)
    with pytest.raises(
        ValueError, match=r"SolverAwareDecision\.observed_campaign_seconds"
    ):
        SolverAwareDecision(
            _error("accepted"),
            _budget(),
            (candidate,),
            observed_campaign_seconds=0.3,
        )


@pytest.mark.parametrize("omitted", ["history/plastic", "materials"])
def test_publication_rejects_undeclared_running_state(omitted: str) -> None:
    history = CompositionEntry(
        jnp.asarray([0.25], dtype=jnp.float64),
        entry_id="history/plastic",
        role="history",
        owner_id="constitutive-law",
        structure_id="history-layout",
        revision_id="history-step-3",
        semantics_id="plastic-strain",
    )
    materials = CompositionEntry(
        jnp.asarray([2.0], dtype=jnp.float64),
        entry_id="materials",
        role="physical-state",
        owner_id="constitutive-law",
        structure_id="material-layout",
        revision_id="material-step-3",
        semantics_id="elastic-modulus",
    )
    composition = Composition((history, materials), boundary_id="accepted-step-3")
    rebind = CompositionRebind(composition, retain=composition.entry_ids)
    obligations = tuple(key for key in composition.entry_ids if key != omitted)
    proofs = tuple(
        pair
        for pair in RouteFeasibility.rebind_dispositions(rebind)
        if pair[0] != omitted
    )
    route = _route("cheap", required=obligations, dispositions=proofs)
    # Self-consistent advertised obligations still cannot omit running state.
    assert route.issues == ()
    with pytest.raises(ValueError, match="accepted composition entries"):
        route.require_rebind(rebind)


@pytest.fixture
def qoi_estimate() -> tuple[DWREstimate, QoICertificate]:
    certificate = QoICertificate(
        "boundary-average-temperature",
        "temperature-H1",
        continuity_bound=2.0,
        frechet_differentiable=True,
        trace_regular=True,
    )
    estimate = DWREstimate(
        (OverlayCellId("thermal-overlay", (0,)), OverlayCellId("thermal-overlay", (1,))),
        np.asarray([0.4, -0.4], dtype=np.float64),
        pollution=(
            ("recovery", 0.03),
            ("curved-boundary", 0.2),
            ("linear-solve", 0.02),
            ("epoch-remap", 0.04),
        ),
        qoi_certificate=certificate,
    )
    return estimate, certificate


_POLLUTION: Mapping[str, DWRPollutionCategory] = {
    "recovery": "field",
    "curved-boundary": "geometry",
    "linear-solve": "algebraic",
    "epoch-remap": "transfer",
}


def test_iga_dwr_cancellation_cannot_hide_absolute_mass_or_pollution(
    qoi_estimate: tuple[DWREstimate, QoICertificate],
) -> None:
    estimate, certificate = qoi_estimate
    actual = estimate.physical_error_evidence(
        "thermal-epoch",
        certificate,
        qoi_id=certificate.qoi_id,
        state_space_id=certificate.state_space_id,
        pollution_categories=_POLLUTION,
    )
    assert estimate.estimate == pytest.approx(0.0)
    assert actual.field_error == pytest.approx(0.83)
    assert actual.geometry_error == pytest.approx(0.2)
    assert actual.algebraic_error == pytest.approx(0.02)
    assert actual.transfer_error == pytest.approx(0.04)
    assert actual.bound == pytest.approx(1.09)
    assert (
        actual.quantity == "qoi"
        and actual.qoi_certificate_id == certificate.certificate_id
    )


@pytest.mark.parametrize(
    "mismatch", ["qoi", "state-space", "certificate", "pollution-owner"]
)
def test_iga_dwr_admission_requires_exact_qoi_and_pollution_ownership(
    qoi_estimate: tuple[DWREstimate, QoICertificate],
    mismatch: str,
) -> None:
    estimate, certificate = qoi_estimate
    supplied = (
        certificate
        if mismatch != "certificate"
        else QoICertificate(
            certificate.qoi_id,
            certificate.state_space_id,
            continuity_bound=3.0,
            frechet_differentiable=True,
            trace_regular=True,
        )
    )
    categories = dict(_POLLUTION)
    if mismatch == "pollution-owner":
        categories.pop("curved-boundary")
    with pytest.raises(ValueError, match="QoI certification|explicit owner"):
        estimate.physical_error_evidence(
            "thermal-epoch",
            supplied,
            qoi_id="different-goal" if mismatch == "qoi" else certificate.qoi_id,
            state_space_id="different-space"
            if mismatch == "state-space"
            else certificate.state_space_id,
            pollution_categories=categories,
        )


def _refined_qoi_decision() -> tuple[SolverAwareDecision, PhysicalErrorEvidence]:
    coarse = QoICertificate(
        "thermal-average",
        "coarse-H1",
        continuity_bound=1.0,
        frechet_differentiable=True,
        trace_regular=True,
    )
    refined = QoICertificate(
        "thermal-average",
        "refined-H1",
        continuity_bound=1.0,
        frechet_differentiable=True,
        trace_regular=True,
    )
    cell = OverlayCellId("thermal-overlay", (0,))

    def evidence(
        revision: str,
        value: float,
        certificate: QoICertificate,
    ) -> PhysicalErrorEvidence:
        estimate = DWREstimate(
            (cell,),
            np.asarray([value], dtype=np.float64),
            pollution=(),
            qoi_certificate=certificate,
        )
        return estimate.physical_error_evidence(
            revision,
            certificate,
            qoi_id=certificate.qoi_id,
            state_space_id=certificate.state_space_id,
            pollution_categories={},
        )

    route = _route("qoi-target")
    baseline = evidence("accepted", 0.8, coarse)
    candidate = SolverAwareCandidate(
        AdaptationAction.H,
        route.candidate_id,
        route,
        evidence(route.target_revision_id, 0.02, refined),
        _candidate("qoi-target", AdaptationAction.H).cost,
    )
    decision = SolverAwareDecision(baseline, _budget(), (candidate,))
    return decision, evidence(route.target_revision_id, 0.025, refined)


def test_refined_qoi_state_space_uses_new_certificate_without_changing_objective() -> (
    None
):
    decision, actual = _refined_qoi_decision()
    selected = decision.selected
    assert selected is not None
    assert decision.baseline.qoi_certificate_id != selected.error.qoi_certificate_id
    assert selected.error.bound <= decision.budget.tolerance
    decision.require_reanalysis(actual)
    assert actual.bound < decision.baseline.bound


@pytest.mark.parametrize("certificate_source", ["accepted-epoch", "foreign-target"])
def test_qoi_reanalysis_rejects_certificate_not_bound_to_selected_target(
    certificate_source: str,
) -> None:
    decision, actual = _refined_qoi_decision()
    certificate_id = (
        decision.baseline.qoi_certificate_id
        if certificate_source == "accepted-epoch"
        else "different-target-qoi-certificate"
    )
    rejected = PhysicalErrorEvidence(
        actual.revision_id,
        actual.objective_id,
        actual.estimator_id,
        field_error=actual.field_error,
        geometry_error=actual.geometry_error,
        algebraic_error=actual.algebraic_error,
        transfer_error=actual.transfer_error,
        quantity="qoi",
        qoi_certificate_id=certificate_id,
    )
    with pytest.raises(ValueError, match="reanalysis-identity-mismatch"):
        decision.require_reanalysis(rejected)


def _metric_proposer() -> LearnedMeshProposer:
    ports = phx.ModelPorts(
        inputs=(full_port("recovered-curvature", (1,)),),
        outputs=(full_port("metric-score", (2, 2)),),
    )
    model = PortedAffine(
        ports,
        out_size=(2, 2),
        weight=jnp.asarray([[-3.0], [5.0], [-1.0], [-2.0]], dtype=jnp.float64),
    )
    return LearnedMeshProposer(
        model,
        kind="metric",
        spatial_dimension=2,
        proposer_id="untrusted-curvature-model",
        ports=ports,
        port_mapping=in_order(ports, ports),
    )


def _planar_source() -> CellMeshingResult:
    return phx.meshing.certify_cell_mesh(
        transfer_triangles(1), phx.SpatialCoordinateContract.si()
    )


def test_learned_features_cannot_be_reused_after_geometry_revision() -> None:
    source = _planar_source()
    features = MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, 0),
        np.ones((source.mesh.coordinates.shape[0], 1), dtype=np.float64),
        feature_ids=("recovered-curvature",),
        feature_owner_id="curvature-recovery",
    )
    moved = phx.meshing.certify_cell_mesh(
        source.mesh.with_coordinates(
            source.mesh.coordinates * 1.01, numeric_version="scaled"
        ),
        source.coordinate_contract,
    )
    assert moved.mesh.topology_id == source.mesh.topology_id
    with pytest.raises(ValueError, match=r"MeshProposalFeatures\.source_result_id"):
        _metric_proposer().propose(moved, features)


def test_learned_indefinite_metric_is_projected_not_promoted_by_model_scores() -> None:
    source = _planar_source()
    features = MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, 0),
        np.ones((source.mesh.coordinates.shape[0], 1), dtype=np.float64),
        feature_ids=("recovered-curvature",),
        feature_owner_id="curvature-recovery",
    )
    proposal = _metric_proposer().propose(source, features)
    assert isinstance(proposal, MeshMetricProposal)
    assert np.all(
        np.linalg.eigvalsh(
            np.asarray(proposal.values) + np.swapaxes(proposal.values, -1, -2)
        )
        < 0.0
    )
    policy = MeshProposalSafetyPolicy(
        source,
        minimum_size=0.2,
        maximum_size=2.0,
        maximum_anisotropy=4.0,
        maximum_displacement=0.1,
    )
    projection = project_mesh_proposal(source, proposal, policy)
    metric, evidence = projection.metric, projection.metric_evidence
    assert metric is not None and evidence is not None
    assert (
        evidence.passed
        and evidence.projected_tensor_count == source.mesh.coordinates.shape[0]
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(metric.values))
    assert np.all(eigenvalues > 0.0)
    assert np.all(eigenvalues[:, -1] / eigenvalues[:, 0] <= 4.0 + 1e-12)
    transaction = prepare_mesh_proposal(source, proposal, policy)
    assert transaction.commit(source, accept=False) is source


def test_learned_coordinate_scores_preserve_protection_and_displacement_budget() -> None:
    source = _planar_source()
    ports = phx.ModelPorts(
        inputs=(full_port("coordinate-features", (3,)),),
        outputs=(full_port("coordinate-target", (2,)),),
    )
    model = PortedAffine(
        ports,
        out_size=(2,),
        weight=jnp.asarray([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float64),
    )
    proposer = LearnedMeshProposer(
        model,
        kind="coordinate",
        spatial_dimension=2,
        proposer_id="untrusted-relocation",
        ports=ports,
        port_mapping=in_order(ports, ports),
    )
    features = MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, 0),
        np.column_stack(
            (
                np.ones(source.mesh.coordinates.shape[0], dtype=np.float64),
                source.mesh.coordinates,
            )
        ),
        feature_ids=("intercept", "coordinate/x", "coordinate/y"),
        feature_owner_id="coordinate-snapshot",
    )
    protected = mesh_proposal_scope(
        source, 0, np.asarray(source.mesh.vertex_global_ids[:1], dtype=np.int64)
    )
    policy = MeshProposalSafetyPolicy(
        source,
        minimum_size=0.1,
        maximum_size=2.0,
        maximum_displacement=0.05,
        protected_scopes=(protected,),
    )
    proposal = proposer.propose(source, features)
    assert isinstance(proposal, MeshCoordinateProposal)
    projection = project_mesh_proposal(source, proposal, policy)
    target = projection.target_coordinates
    assert target is not None
    np.testing.assert_array_equal(target[0], source.mesh.coordinates[0])
    displacement = np.linalg.norm(np.asarray(target - source.mesh.coordinates), axis=1)
    np.testing.assert_allclose(
        displacement[1:], 0.05, rtol=0.0, atol=_LEARNED_DISPLACEMENT_ATOL
    )


def test_untrusted_marking_uses_explicit_family_policy_without_simplex_substitution() -> (
    None
):
    points = np.asarray(
        tuple(
            (float(x), float(y), float(z))
            for z in range(2)
            for y in range(2)
            for x in range(2)
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "hexes",
                "hexahedron",
                jnp.asarray(((0, 1, 3, 2, 4, 5, 7, 6),), dtype=jnp.int32),
                global_ids=jnp.asarray((10,), dtype=jnp.int64),
            ),
        ),
    )
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    proposal = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 3),
        np.ones(1, dtype=np.float64),
        proposer_id="untrusted-family-marker",
    )
    safety = MeshProposalSafetyPolicy(
        source,
        minimum_size=0.1,
        maximum_size=2.0,
        maximum_displacement=0.1,
    )
    with pytest.raises(ValueError, match=r"MeshProposal\.native_policy"):
        prepare_mesh_proposal(source, proposal, safety)
    prepared = prepare_mesh_proposal(
        source,
        proposal,
        safety,
        native_policy=MeshAdaptationPolicy(
            MeshAdaptationRoute.NATIVE_MIXED,
            limits=safety.limits,
            audit_policy=safety.audit_policy,
        ),
    )
    accepted = prepared.commit(source)
    adaptation = prepared.adaptation
    assert adaptation is not None and adaptation.transfer is not None
    assert adaptation.route is MeshAdaptationRoute.NATIVE_MIXED
    assert accepted.audit.entity_counts[-1] == 8
    assert all(block.cell_kind == "hexahedron" for block in accepted.mesh.blocks)
    coarse = (
        mesh.coordinates[:, 0] + 2.0 * mesh.coordinates[:, 1] - mesh.coordinates[:, 2]
    )
    expected = (
        accepted.mesh.coordinates[:, 0]
        + 2.0 * accepted.mesh.coordinates[:, 1]
        - accepted.mesh.coordinates[:, 2]
    )
    np.testing.assert_allclose(
        adaptation.transfer.apply(coarse), expected, rtol=0.0, atol=_FAMILY_TRANSFER_ATOL
    )
    assert prepared.commit(source, accept=False) is source


def test_native_hp_hierarchy_cannot_repair_a_new_source_geometry_revision() -> None:
    mesh = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        (
            CellBlock(
                "quads",
                "quadrilateral",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
                global_ids=jnp.asarray((10,), dtype=jnp.int64),
            ),
        ),
    )
    topology, geometry = phx.discretization.fem.initial_finite_element_hp_topology(
        mesh, (1, 1), 5
    )
    epoch = phx.discretization.fem.prepare_finite_element_hp_epoch(
        topology, geometry, "u"
    )
    moved = epoch.mesh.with_coordinates(
        epoch.mesh.coordinates + 0.125, numeric_version="translated"
    )
    source = phx.meshing.certify_cell_mesh(moved, phx.SpatialCoordinateContract.si())
    proposal = MeshMarkingProposal(
        source,
        mesh_proposal_scope(source, 2),
        np.ones(1, dtype=np.float64),
        proposer_id="untrusted-hp-marker",
    )
    safety = MeshProposalSafetyPolicy(
        source,
        minimum_size=0.1,
        maximum_size=2.0,
        maximum_displacement=0.1,
    )
    assert source.mesh.topology_id == epoch.mesh.topology_id
    assert source.mesh.geometry_id != epoch.mesh.geometry_id
    with pytest.raises(ValueError, match=r"MeshAdaptation\.request\.hierarchy"):
        prepare_mesh_proposal(
            source,
            proposal,
            safety,
            hierarchy=epoch,
            native_policy=MeshAdaptationPolicy(
                MeshAdaptationRoute.HP,
                limits=safety.limits,
                audit_policy=safety.audit_policy,
            ),
        )


@dataclass(frozen=True)
class _NativeCase:
    source: CellMeshingResult
    adaptation: MeshAdaptationResult
    source_space: FiniteElementDiscretization
    target_space: FiniteElementDiscretization
    field: FiniteElementFieldSpec
    transfer: FiniteElementFieldTransfer


def _native_case(dimension: int) -> _NativeCase:
    mesh = (
        transfer_triangles(1) if dimension == 2 else kuhn_cube(1, orientation_swap=(0, 1))
    )
    source = curved_shear_source(mesh, 2, shear_term(2, dimension))
    if not isinstance(source, CellMeshingResult):
        raise TypeError(
            "curved_shear_source.result must be a CellMeshingResult; got a different result kind."
        )
    adaptation = bisect(source, (0,))
    if not isinstance(adaptation, MeshAdaptationResult):
        raise TypeError(
            "bisect.result must be a MeshAdaptationResult; got a different result kind."
        )
    field = FiniteElementFieldSpec(
        "u", lagrange_element("triangle" if dimension == 2 else "tetrahedron", 2)
    )
    source_space = FiniteElementPlan(
        source.mesh, field, coordinate_spec=source.geometry
    ).prepare()
    target_space = FiniteElementPlan(
        adaptation.target.mesh,
        field,
        coordinate_spec=adaptation.target.geometry,
    ).prepare()
    transfer = prepare_nested_field_transfer(
        source_space,
        target_space,
        adaptation.parent_cells,
        field_name="u",
        parent_reference_vertices=adaptation.parent_reference_vertices,
    )
    return _NativeCase(source, adaptation, source_space, target_space, field, transfer)


@pytest.fixture(scope="module")
def native_3d() -> _NativeCase:
    return _native_case(3)


def _physical_field(points: Array, args: object) -> Array:
    del args
    # This polynomial is reproduced in parent reference space under the shear.
    return 1.0 + points[..., 0] + points[..., 1] * points[..., -1]


@pytest.mark.parametrize(
    "dimension", [2, 3], ids=["curved-triangles", "curved-tetrahedra"]
)
def test_native_curved_transfer_reproduces_field_on_independent_coordinate_map(
    dimension: int,
    native_3d: _NativeCase,
) -> None:
    case = _native_case(2) if dimension == 2 else native_3d
    original = case.source_space.project("u", _physical_field)
    expected = case.target_space.project("u", _physical_field)
    actual = case.transfer.transfer.apply(original)
    assert case.adaptation.status.converged and case.transfer.evidence.passed
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=_PIPELINE_VALUE_ATOL)
    dual = jnp.linspace(-0.7, 1.3, actual.size, dtype=jnp.float64)
    np.testing.assert_allclose(
        jnp.vdot(actual, dual),
        jnp.vdot(original, case.transfer.transfer.pullback(dual)),
        rtol=0.0,
        atol=_PIPELINE_VALUE_ATOL,
    )
    assert not case.transfer.transfer.conservative
    geometry_transition = case.adaptation.geometry_transition
    assert geometry_transition is not None and geometry_transition.evidence.exact


def _native_trial(
    case: _NativeCase,
) -> tuple[
    FiniteElementAcceptedState,
    FiniteElementTopologyTransaction,
    SolverAwareDecision,
    CompositionRebind,
]:
    material = MaterialTransaction(
        (
            MaterialState(
                MaterialSiteId("plastic-history"),
                "plastic-history",
                jnp.linspace(
                    0.1, 0.6, case.source.mesh.blocks[0].cell_count, dtype=jnp.float64
                ),
            ),
        )
    )
    accepted = FiniteElementAcceptedState(
        (case.source_space.project("u", _physical_field),),
        jnp.asarray(0.5, dtype=jnp.float64),
        3,
        case.source.mesh.topology_id,
        case.source_space.prepared_id,
        "source-compiled-layout",
        materials=material,
        schedule_cursor=2,
        state_version=3,
        transition_id="previous-receipt",
    )
    captured: list[CompositionRebind] = []

    def capture(rebind: CompositionRebind) -> CompositionRebind:
        captured.append(rebind)
        return rebind

    def remap_materials(
        value: MaterialTransaction, lineage: object, args: object
    ) -> MaterialTransaction:
        del lineage, args
        before = value.state("plastic-history")
        remapped = before.committed[case.adaptation.parent_cells]
        return MaterialTransaction(
            (
                MaterialState(
                    before.site_id,
                    before.model_id,
                    remapped,
                    state_version=before.state_version + 1,
                ),
            )
        )

    def certify(
        mesh: CellMesh,
        fields: tuple[Array, ...],
        materials: MaterialTransaction | None,
        lineage: object,
        args: object,
    ) -> bool:
        del mesh, fields, materials, lineage, args
        return True

    def reject_trial(
        mesh: CellMesh,
        fields: tuple[Array, ...],
        materials: MaterialTransaction | None,
        lineage: object,
        args: object,
    ) -> bool:
        del mesh, fields, materials, lineage, args
        return False

    # A native prepared trial provides composition/transport identities.
    trial = FiniteElementTopologyTransaction(
        reject_trial,
        fields=case.field,
        material_transfer=remap_materials,
        composition_rebind=capture,
    )
    trial_result = trial.execute(accepted, case.source.mesh, case.adaptation)
    assert not bool(trial_result.committed) and trial_result.receipt is not None
    core = captured[0]
    candidate_id = case.adaptation.result_id
    target = case.adaptation.target
    route = RouteFeasibility(
        case.source.result_id,
        target.result_id,
        case.adaptation.route.value,
        candidate_id,
        cell_families=tuple(block.cell_kind for block in target.mesh.blocks),
        geometry_layout_id=target.geometry.geometry_layout_id,
        field_layouts=tuple(
            (space.name, space.layout.layout_id)
            for space in case.target_space.field_spaces
        ),
        compiled_layout_id="target-compiled-layout",
        geometry_certificate_id=target.audit.report_id,
        topology_certificate_id=target.audit.report_id,
        required_state_ids=core.source.entry_ids,
        state_dispositions=RouteFeasibility.rebind_dispositions(core),
        dofs=sum(space.layout.size for space in case.target_space.field_spaces),
        condition_estimate=5.0,
    )
    candidate = _candidate(candidate_id, AdaptationAction.H, feasibility=route)
    decision = SolverAwareDecision(
        _error(case.source.result_id, field=0.8), _budget(), (candidate,)
    )
    executor = FiniteElementTopologyTransaction(
        certify,
        fields=case.field,
        material_transfer=remap_materials,
    )
    return accepted, executor, decision, core


@pytest.mark.parametrize(
    "failure", ["physical-target", "same-estimator", "wrong-objective"]
)
def test_independent_native_reanalysis_rejection_retains_accepted_objects_and_receipt(
    native_3d: _NativeCase,
    failure: str,
) -> None:
    accepted, executor, decision, core = _native_trial(native_3d)
    selected = decision.selected
    assert selected is not None

    def reanalysis(
        mesh: CellMesh,
        fields: tuple[Array, ...],
        materials: MaterialTransaction | None,
        args: object,
    ) -> PhysicalErrorEvidence:
        del args
        assert mesh is native_3d.adaptation.target.mesh
        np.testing.assert_allclose(
            fields[0],
            native_3d.target_space.project("u", _physical_field),
            atol=_PIPELINE_VALUE_ATOL,
        )
        assert materials is not None
        return _error(
            native_3d.adaptation.target.result_id,
            field=0.2 if failure == "physical-target" else 0.01,
            estimator=selected.error.estimator_id
            if failure == "same-estimator"
            else "independent-owner",
            objective="wrong-goal" if failure == "wrong-objective" else "energy-error",
        )

    result = executor.execute(
        accepted,
        native_3d.source.mesh,
        native_3d.adaptation,
        decision=decision,
        reanalysis=reanalysis,
        compiled_layout_id="target-compiled-layout",
    )
    assert not bool(result.committed)
    assert result.state is accepted and result.mesh is native_3d.source.mesh
    assert result.state.fields[0] is accepted.fields[0]
    assert result.state.materials is accepted.materials
    assert (
        result.state.transition_id == "previous-receipt"
        and result.state.state_version == 3
    )
    receipt = result.receipt
    assert receipt is not None and not receipt.published and not receipt.boundary_accepted
    assert all(receipt.transport_accepted)
    assert receipt.composition.composition_id == core.source.composition_id
    assert set(receipt.remapped) == {"field/u", "materials"}
    assert result.adaptation is None


@pytest.mark.parametrize(
    "accept_reanalysis",
    [True, False],
    ids=["publish-solved-state", "reject-solved-state"],
)
@pytest.mark.parametrize(
    "semantics",
    [None, "conservative-density", "intensive", "material-compatible"],
    ids=["undeclared-content", "density-content", "intensive", "material-compatible"],
)
def test_native_remap_then_pde_refresh_publishes_only_reanalyzed_state(
    native_3d: _NativeCase,
    accept_reanalysis: bool,
    semantics: SourceRealizationFieldSemantics | None,
) -> None:
    case = native_3d
    source_values = case.source_space.field_spaces[0].vector_space.zeros()
    accepted = FiniteElementAcceptedState(
        (source_values,),
        0.0,
        0,
        case.source.mesh.topology_id,
        case.source_space.prepared_id,
        "accepted-reaction-layout",
    )
    form = phx.equations.FiniteElementForm(
        "reaction-refresh",
        "u",
        (
            phx.equations.MassAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    _physical_field,
                    coefficient_id="manufactured-reaction-field",
                ),
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(form, case.target_space)
    system, load = compiled.linear_system()
    solved = phx.linalg.solve(
        system, load, policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    )
    assert bool(jnp.all(solved.successful))
    coefficients = compiled.expand(solved.value)
    expected = case.target_space.project("u", _physical_field)
    np.testing.assert_allclose(
        coefficients, expected, rtol=0.0, atol=_REANALYZED_FIELD_ATOL
    )

    def refresh(
        carried: tuple[Array, ...],
        adaptation: MeshAdaptationResult,
        args: object,
    ) -> tuple[Array, ...]:
        del args
        assert adaptation.result_id == case.adaptation.result_id
        np.testing.assert_array_equal(carried[0], jnp.zeros_like(expected))
        return (coefficients,)

    def certify(
        mesh: CellMesh,
        fields: tuple[Array, ...],
        materials: MaterialTransaction | None,
        lineage: object,
        args: object,
    ) -> bool:
        del mesh, materials, lineage, args
        np.testing.assert_allclose(
            fields[0], expected, rtol=0.0, atol=_REANALYZED_FIELD_ATOL
        )
        return accept_reanalysis

    result = FiniteElementTopologyTransaction(
        certify,
        fields=case.field,
        field_transfer=refresh,
        surface_chart_semantics=None if semantics is None else {"u": semantics},
    ).execute(accepted, case.source.mesh, case.adaptation)
    assert result.receipt is not None
    # This P2 nested field transfer carries no physical content ledger.
    # The generic PDE re-solve may advance its field, but an explicit density
    # declaration must not promote interpolation to conserved material content.
    assert not result.transfers[0].transfer.conservative
    if accept_reanalysis and semantics != "conservative-density":
        assert bool(result.committed) and result.receipt.published
        np.testing.assert_allclose(
            result.state.fields[0], expected, rtol=0.0, atol=_REANALYZED_FIELD_ATOL
        )
        np.testing.assert_array_equal(accepted.fields[0], source_values)
    else:
        assert not bool(result.committed) and not result.receipt.published
        assert result.state is accepted and result.mesh is case.source.mesh
        assert result.diagnostics == (
            "transfer-evidence-rejected"
            if semantics == "conservative-density"
            else "candidate-certification-rejected"
        )
        if semantics == "conservative-density":
            assert not all(result.receipt.transport_accepted)
        else:
            assert all(result.receipt.transport_accepted)


def test_native_independent_reanalysis_publishes_one_atomic_epoch(
    native_3d: _NativeCase,
) -> None:
    accepted, executor, decision, _ = _native_trial(native_3d)

    def reanalysis(
        mesh: CellMesh,
        fields: tuple[Array, ...],
        materials: MaterialTransaction | None,
        args: object,
    ) -> PhysicalErrorEvidence:
        del mesh, fields, materials, args
        return _error(
            native_3d.adaptation.target.result_id, estimator="independent-owner"
        )

    result = executor.execute(
        accepted,
        native_3d.source.mesh,
        native_3d.adaptation,
        decision=decision,
        reanalysis=reanalysis,
        compiled_layout_id="target-compiled-layout",
    )
    receipt = result.receipt
    assert bool(result.committed) and receipt is not None and receipt.published
    assert result.state.state_version == accepted.state_version + 1
    assert result.state.transition_id == receipt.receipt_id
    assert result.state.schedule_cursor == accepted.schedule_cursor
    np.testing.assert_allclose(
        result.state.fields[0],
        native_3d.target_space.project("u", _physical_field),
        atol=_PIPELINE_VALUE_ATOL,
    )
    assert accepted.materials is not None and result.state.materials is not None
    np.testing.assert_allclose(
        result.state.materials.state("plastic-history").committed,
        accepted.materials.state("plastic-history").committed[
            native_3d.adaptation.parent_cells
        ],
        atol=0.0,
    )


@pytest.mark.parametrize("mismatch", ["field-order", "geometry-order", "compiled-layout"])
def test_native_admission_rejects_mislabeled_actions_and_foreign_compiler_layout(
    native_3d: _NativeCase,
    mismatch: str,
) -> None:
    accepted, executor, decision, _ = _native_trial(native_3d)
    selected = decision.selected
    assert selected is not None
    if mismatch != "compiled-layout":
        action = (
            AdaptationAction.P
            if mismatch == "field-order"
            else AdaptationAction.GEOMETRY_ORDER
        )
        mislabeled = SolverAwareCandidate(
            action,
            selected.candidate_id,
            selected.feasibility,
            selected.error,
            selected.cost,
        )
        decision = SolverAwareDecision(decision.baseline, decision.budget, (mislabeled,))

    def reanalysis(
        mesh: CellMesh,
        fields: tuple[Array, ...],
        materials: MaterialTransaction | None,
        args: object,
    ) -> PhysicalErrorEvidence:
        del mesh, fields, materials, args
        return _error(
            native_3d.adaptation.target.result_id, estimator="independent-owner"
        )

    expected_field = (
        r"SolverAwareDecision\.selected\.feasibility"
        if mismatch == "compiled-layout"
        else r"SolverAwareDecision\.selected must match action="
    )
    with pytest.raises(ValueError, match=expected_field):
        executor.execute(
            accepted,
            native_3d.source.mesh,
            native_3d.adaptation,
            decision=decision,
            reanalysis=reanalysis,
            compiled_layout_id="foreign-layout"
            if mismatch == "compiled-layout"
            else "target-compiled-layout",
        )


def test_native_decision_requires_independent_reanalysis_before_publication(
    native_3d: _NativeCase,
) -> None:
    accepted, executor, decision, _ = _native_trial(native_3d)
    result = executor.execute(
        accepted,
        native_3d.source.mesh,
        native_3d.adaptation,
        decision=decision,
        compiled_layout_id="target-compiled-layout",
    )
    assert not bool(result.committed) and result.receipt is None
    assert result.state is accepted and result.mesh is native_3d.source.mesh
    assert result.diagnostics == "independent-reanalysis-required"


@dataclass(frozen=True)
class _DerivativeCase:
    pipeline: Callable[[Array], Array]
    reference: Callable[[float], tuple[NDArray[np.float64], NDArray[np.float64]]]
    adjoint: Callable[[float, NDArray[np.float64]], float]
    evidence: FixedEpochDerivativeEvidence
    margins: tuple[tuple[str, float], ...]
    qualifications: tuple[DesignQualificationEvidence, ...]


@pytest.fixture(scope="module")
def fixed_pipeline(native_3d: _NativeCase) -> _DerivativeCase:
    space = native_3d.source_space
    coordinates = space.default_runtime.coordinates
    runtime = space.default_runtime
    mass, stiffness = space.assemble_field_operators("u", runtime)
    base_mass = np.asarray(mass.as_dense(), dtype=np.float64)
    base_stiffness = np.asarray(stiffness.as_dense(), dtype=np.float64)
    count = base_mass.shape[0]
    forcing = np.linspace(0.2, 1.1, count, dtype=np.float64)
    transfer_matrix = np.asarray(
        native_3d.transfer.transfer.apply(jnp.eye(count, dtype=jnp.float64)),
        dtype=np.float64,
    )

    def pipeline(scale: Array) -> Array:
        realized = space.prepare_runtime(
            coordinates * scale, numeric_version="uniform-dilation"
        )
        mass_, stiffness_ = space.assemble_field_operators("u", realized)
        matrix = stiffness_.as_dense() + mass_.as_dense()
        load = mass_.mv(jnp.asarray(forcing, dtype=jnp.float64))
        return native_3d.transfer.transfer.apply(jnp.linalg.solve(matrix, load))

    def reference(scale: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # In 3D, uniform dilation gives K(s)=s K(1), M(s)=s^3 M(1).
        # This host sensitivity solve is independent of autodiff and runtime geometry.
        matrix = scale * base_stiffness + scale**3 * base_mass
        load = scale**3 * (base_mass @ forcing)
        solution = np.linalg.solve(matrix, load)
        derivative = np.linalg.solve(
            matrix,
            3.0 * scale**2 * (base_mass @ forcing)
            - (base_stiffness + 3.0 * scale**2 * base_mass) @ solution,
        )
        return transfer_matrix @ solution, transfer_matrix @ derivative

    def adjoint(scale: float, dual: NDArray[np.float64]) -> float:
        matrix = scale * base_stiffness + scale**3 * base_mass
        load = scale**3 * (base_mass @ forcing)
        solution = np.linalg.solve(matrix, load)
        multiplier = np.linalg.solve(matrix.T, transfer_matrix.T @ dual)
        sensitivity_load = (
            3.0 * scale**2 * (base_mass @ forcing)
            - (base_stiffness + 3.0 * scale**2 * base_mass) @ solution
        )
        return float(multiplier @ sensitivity_load)

    scale = jnp.asarray(1.2, dtype=jnp.float64)
    value, tangent = jax.jvp(pipeline, (scale,), (jnp.asarray(1.0, dtype=jnp.float64),))
    expected, sensitivity = reference(1.2)
    np.testing.assert_allclose(
        value, expected, rtol=_PIPELINE_VALUE_RTOL, atol=_PIPELINE_VALUE_ATOL
    )
    np.testing.assert_allclose(
        tangent,
        sensitivity,
        rtol=_PIPELINE_DERIVATIVE_RTOL,
        atol=_PIPELINE_DERIVATIVE_ATOL,
    )
    gradient_error = float(np.max(np.abs(np.asarray(tangent) - sensitivity)))
    coercivity = float(
        np.min(np.linalg.eigvalsh(1.2 * base_stiffness + 1.2**3 * base_mass))
    )
    parent_centers = np.mean(native_3d.adaptation.parent_reference_vertices, axis=1)
    barycentric = np.concatenate(
        (1.0 - np.sum(parent_centers, axis=1, keepdims=True), parent_centers), axis=1
    )
    donor_margin = float(np.min(barycentric))
    margins = (
        ("geometry-singularity", 1.2),
        ("pde-coercivity", coercivity),
        ("donor-switch", donor_margin),
    )
    plans = (
        native_3d.source.geometry.geometry_layout_id,
        space.prepared_id,
        native_3d.transfer.transfer_id,
    )
    qualifications = tuple(
        DesignQualificationEvidence(
            DerivativeTier.Q1,
            plan,
            native_3d.source.result_id,
            plan,
            gradient_error=gradient_error,
            event_ids=(event,),
            event_margins=(margin,),
            valid=True,
        )
        for plan, (event, margin) in zip(plans, margins, strict=True)
    )
    evidence = FixedEpochDerivativeEvidence(
        native_3d.source.result_id,
        "uniform-dilation",
        geometry=qualifications[0],
        pde=qualifications[1],
        transfer=qualifications[2],
        routes=tuple(zip(("geometry", "pde", "transfer"), plans, strict=True)),
    )
    return _DerivativeCase(
        pipeline, reference, adjoint, evidence, margins, qualifications
    )


def test_fixed_route_geometry_pde_transfer_jvp_matches_independent_sensitivity(
    fixed_pipeline: _DerivativeCase,
) -> None:
    case = fixed_pipeline
    case.evidence.require_valid(
        case.evidence.epoch_id,
        case.evidence.parameter_id,
        routes=case.evidence.routes,
        event_margins=case.margins,
    )
    scale, direction = (
        jnp.asarray(1.2, dtype=jnp.float64),
        jnp.asarray(0.37, dtype=jnp.float64),
    )
    _, tangent = jax.jvp(case.pipeline, (scale,), (direction,))
    _, expected = case.reference(1.2)
    np.testing.assert_allclose(
        tangent,
        0.37 * expected,
        rtol=_PIPELINE_DERIVATIVE_RTOL,
        atol=_PIPELINE_DERIVATIVE_ATOL,
    )
    step = 1e-5
    finite_difference = (
        case.pipeline(scale + step * direction) - case.pipeline(scale - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        tangent,
        finite_difference,
        rtol=_PIPELINE_DIFFERENCE_RTOL,
        atol=_PIPELINE_DIFFERENCE_ATOL,
    )


def test_fixed_route_geometry_pde_transfer_vjp_matches_independent_adjoint(
    fixed_pipeline: _DerivativeCase,
) -> None:
    case = fixed_pipeline
    scale = jnp.asarray(1.2, dtype=jnp.float64)
    value, pullback = jax.vjp(case.pipeline, scale)
    dual = jnp.linspace(-0.8, 0.9, value.size, dtype=jnp.float64)
    expected = case.adjoint(1.2, np.asarray(dual, dtype=np.float64))
    np.testing.assert_allclose(
        pullback(dual)[0],
        expected,
        rtol=_PIPELINE_DERIVATIVE_RTOL,
        atol=_PIPELINE_DERIVATIVE_ATOL,
    )
    direction = jnp.asarray(0.37, dtype=jnp.float64)
    _, tangent = jax.jvp(case.pipeline, (scale,), (direction,))
    np.testing.assert_allclose(
        jnp.vdot(dual, tangent),
        jnp.vdot(pullback(dual)[0], direction),
        rtol=_PIPELINE_VALUE_RTOL,
        atol=_PIPELINE_VALUE_ATOL,
    )


@pytest.mark.parametrize(
    "event",
    [
        "zero-margin",
        "donor-switch",
        "retopology",
        "acceptance",
        "epoch-change",
        "parameter-change",
    ],
)
def test_fixed_route_derivative_invalidates_discrete_events_instead_of_smoothing_them(
    fixed_pipeline: _DerivativeCase,
    event: str,
) -> None:
    case = fixed_pipeline
    routes = case.evidence.routes
    margins = case.margins
    topology_event = None
    epoch = "new-epoch" if event == "epoch-change" else case.evidence.epoch_id
    parameter = (
        "different-physical-parameter"
        if event == "parameter-change"
        else case.evidence.parameter_id
    )
    if event == "zero-margin":
        margins = tuple(
            (name, 0.0 if name == "donor-switch" else value) for name, value in margins
        )
    elif event == "donor-switch":
        routes = (*routes, ("donor", "new-donor-stencil"))
    elif event in ("retopology", "acceptance"):
        topology_event = event
    with pytest.raises(
        ValueError,
        match="event-boundary|fixed-route-changed|stopped-topology-event|epoch-or-parameter-changed",
    ):
        case.evidence.require_valid(
            epoch,
            parameter,
            routes=routes,
            event_margins=margins,
            topology_event_id=topology_event,
        )


def test_fixed_route_derivative_rejects_wrong_stage_qualification(
    fixed_pipeline: _DerivativeCase,
) -> None:
    case = fixed_pipeline
    unrelated_pde = replace(
        case.qualifications[1], execution_plan_id="different-pde-discretization"
    )
    with pytest.raises(ValueError, match="qualified execution plans"):
        FixedEpochDerivativeEvidence(
            case.evidence.epoch_id,
            case.evidence.parameter_id,
            geometry=case.qualifications[0],
            pde=unrelated_pde,
            transfer=case.qualifications[2],
            routes=case.evidence.routes,
        )


def test_geometry_realization_invalidates_fixed_epoch_derivatives_without_retopology(
    fixed_pipeline: _DerivativeCase,
) -> None:
    case = fixed_pipeline
    issues = case.evidence.invalidation_issues(
        case.evidence.epoch_id,
        case.evidence.parameter_id,
        routes=case.evidence.routes,
        event_margins=case.margins,
        geometry_realization_id="actual-changed-coordinate-map",
    )
    assert "stopped-geometry-realization-event" in issues
    assert "stopped-topology-event" not in issues
    with pytest.raises(ValueError, match="stopped-geometry-realization-event"):
        case.evidence.require_valid(
            case.evidence.epoch_id,
            case.evidence.parameter_id,
            routes=case.evidence.routes,
            event_margins=case.margins,
            geometry_realization_id="actual-changed-coordinate-map",
        )


@pytest.mark.parametrize("stage", [0, 1, 2], ids=["geometry", "pde", "transfer"])
def test_fixed_route_derivative_rejects_foreign_numeric_epoch(
    fixed_pipeline: _DerivativeCase,
    stage: int,
) -> None:
    case = fixed_pipeline
    qualifications = list(case.qualifications)
    qualifications[stage] = replace(
        qualifications[stage], numeric_revision_id="foreign-epoch"
    )
    with pytest.raises(ValueError, match="numeric epoch"):
        FixedEpochDerivativeEvidence(
            case.evidence.epoch_id,
            case.evidence.parameter_id,
            geometry=qualifications[0],
            pde=qualifications[1],
            transfer=qualifications[2],
            routes=case.evidence.routes,
        )


def _run_hp_campaign(p_degree: int, /) -> dict[str, Any]:
    return run_on_forced_devices(
        "tests._support.solver_hp_campaign",
        "hp_campaign",
        (str(p_degree),),
        devices=1,
        env={},
    )


def _assert_hp_campaign(result: Mapping[str, Any]) -> None:
    assert result["accepted_L2_error"] < result["tolerance"]
    assert result["accepted_L2_error"] < result["baseline_L2_error"]
    assert result["transfer_integral_error"] < _CAMPAIGN_TRANSFER_RESIDUAL
    assert set(result["history_integral_errors"]) == {
        "previous-reaction-state",
        "accumulated-reaction-state",
    }
    assert all(
        error < _CAMPAIGN_TRANSFER_RESIDUAL
        for error in result["history_integral_errors"].values()
    )
    assert {trial["action"] for trial in result["trials"]} == {"h", "p"}
    assert all(sum(trial["phase_seconds"]) > 0.0 for trial in result["trials"])
    for trial in result["trials"]:
        assert (
            max(
                trial["transfer_derivatives"][name]
                for name in (
                    "JVP_error",
                    "VJP_error",
                    "duality_error",
                )
            )
            < _CAMPAIGN_TRANSFER_RESIDUAL
        )


def _assert_metric_campaign(result: Mapping[str, Any]) -> None:
    assert result["accepted_L2_error"] < result["tolerance"]
    assert result["accepted_L2_error"] < result["baseline_L2_error"]
    assert result["integral_error"] < _CAMPAIGN_TRANSFER_RESIDUAL
    assert result["transfer_duality_error"] < _CAMPAIGN_TRANSFER_RESIDUAL
    assert (
        result["transfer_JVP_finite_difference_error"]
        < _METRIC_TRANSFER_DIFFERENCE_RESIDUAL
    )
    assert (
        result["transfer_VJP_finite_difference_error"]
        < _METRIC_TRANSFER_DIFFERENCE_RESIDUAL
    )
    assert result["physical_reanalysis_rollback"]


@pytest.fixture(scope="module")
def hp_campaign_reports() -> dict[int, dict[str, Any]]:
    return {degree: _run_hp_campaign(degree) for degree in (2, 3)}


@pytest.fixture(scope="module")
def metric_campaign_report() -> dict[str, Any]:
    return run_metric()


@pytest.mark.parametrize("p_degree", [2, 3], ids=["quadratic-field", "cubic-field"])
def test_prepared_hp_physical_campaign_transports_named_field_histories(
    p_degree: int,
    hp_campaign_reports: dict[int, dict[str, Any]],
) -> None:
    _assert_hp_campaign(hp_campaign_reports[p_degree])


def test_native_metric_physical_campaign_accepts_solve_and_rolls_back_rejected_reanalysis(
    metric_campaign_report: dict[str, Any],
) -> None:
    _assert_metric_campaign(metric_campaign_report)


def test_missing_native_engine_cannot_publish_a_learned_metric_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax import _meshcore

    # Planar metric adaptation is JAX-owned and does not require this
    # binary. The 3D route constructs TetMesh3D and must refuse its absence.
    source = phx.meshing.certify_cell_mesh(
        kuhn_cube(1, orientation_swap=(0, 1)), phx.SpatialCoordinateContract.si()
    )
    features = MeshProposalFeatures(
        source,
        mesh_proposal_scope(source, 0),
        np.ones((source.mesh.coordinates.shape[0], 1)),
        feature_ids=("recovered-curvature",),
        feature_owner_id="prepared-reaction-estimator",
    )
    ports = phx.ModelPorts(
        inputs=(full_port("recovered-curvature", (1,)),),
        outputs=(full_port("metric-score", (3, 3)),),
    )
    model = PortedAffine(
        ports,
        out_size=(3, 3),
        weight=(jnp.eye(3, dtype=jnp.float64) * 16.0).reshape(9, 1),
    )
    proposer = LearnedMeshProposer(
        model,
        kind="metric",
        spatial_dimension=3,
        proposer_id="untrusted-volume-curvature-model",
        ports=ports,
        port_mapping=in_order(ports, ports),
    )
    proposal = proposer.propose(source, features)
    policy = MeshProposalSafetyPolicy(
        source,
        minimum_size=0.1,
        maximum_size=2.0,
        maximum_displacement=0.1,
    )
    source_coordinates = np.asarray(source.mesh.coordinates).copy()
    monkeypatch.setattr(
        _meshcore, "_current", lambda: "Native engine deliberately absent."
    )
    with pytest.raises(_meshcore.MeshcoreUnavailableError):
        prepare_mesh_proposal(source, proposal, policy)
    np.testing.assert_array_equal(source.mesh.coordinates, source_coordinates)


def _assert_restored_hp_campaign(
    authored: Mapping[str, Any], restored: Mapping[str, Any]
) -> None:
    assert restored["accepted_L2_error"] < authored["tolerance"]
    assert restored["independently_recomputed_L2_error"] < authored["tolerance"]
    assert set(restored["history_integral_errors"]) == set(
        authored["history_integral_errors"]
    )
    assert all(
        error < _CAMPAIGN_TRANSFER_RESIDUAL
        for error in restored["history_integral_errors"].values()
    )


@pytest.fixture(scope="module")
def archived_hp_campaign(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], dict[str, Any]]:
    archive = tmp_path_factory.mktemp("hp-campaign")
    authored = run(p_degree=3, with_history=True, archive=archive)
    return authored, resume_hp(archive)


def test_hp_physical_campaign_archive_rebuilds_fields_and_integrator_histories(
    archived_hp_campaign: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _assert_restored_hp_campaign(*archived_hp_campaign)


class _NativeMetricFVCase(NamedTuple):
    adaptation: MeshAdaptationResult
    previous: NativeExecutionRecord
    source: UnstructuredFiniteVolumeDiscretization
    target: UnstructuredFiniteVolumeDiscretization


@pytest.fixture(scope="module")
def native_metric_fv_owners() -> _NativeMetricFVCase:
    """Return a tetrahedral native event for resource accounting."""
    from phydrax.discretization import UnstructuredFiniteVolumePlan
    from phydrax.meshing import MeshingLimits
    from phydrax.meshing._tetra_metric import MetricRemeshingEvidence

    points, tetrahedra = kuhn_cube_arrays(orientation_swap=(0, 1))
    points *= np.asarray((2.0, 0.5, 1.0), dtype=np.float64)
    mesh = CellMesh(points, (CellBlock("box", "tetrahedron", tetrahedra),))
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    limits = MeshingLimits(
        maximum_work_units=640_000,
        maximum_geometry_queries=1_280_000,
        maximum_scratch_bytes=81_920_000,
    )
    proposal = MeshMetricProposal(
        source,
        mesh_proposal_scope(source, 0),
        np.broadcast_to(
            np.diag((0.25, 4.0, 1.0)), (source.mesh.coordinates.shape[0], 3, 3)
        ),
        proposer_id="resource-contract-tetrahedral-metric",
    )
    safety = MeshProposalSafetyPolicy(
        source,
        minimum_size=0.1,
        maximum_size=2.0,
        maximum_displacement=0.1,
        maximum_anisotropy=20.0,
        maximum_gradation=2.0,
        limits=limits,
    )
    trial = prepare_mesh_proposal(
        source,
        proposal,
        safety,
        native_policy=MeshAdaptationPolicy(
            MeshAdaptationRoute.NATIVE_METRIC_3D,
            limits=limits,
            maximum_passes=12,
            relocation=False,
        ),
    )
    adaptation = trial.adaptation
    assert adaptation is not None
    assert isinstance(adaptation.evidence, MetricRemeshingEvidence)
    previous = adaptation.evidence.execution_evidence
    assert previous is not None
    previous.require_valid()
    assert adaptation.status.converged and trial.admissible
    old = UnstructuredFiniteVolumePlan.from_cell_mesh(source.mesh).prepare(
        cell_geometry=source.geometry
    )
    new = UnstructuredFiniteVolumePlan.from_cell_mesh(adaptation.target.mesh).prepare(
        cell_geometry=adaptation.target.geometry
    )
    return _NativeMetricFVCase(adaptation, previous, old, new)


def test_native_remap_authenticated_root_retains_owners_and_physical_content(
    native_metric_fv_owners: _NativeMetricFVCase,
) -> None:
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.discretization.finite_volume._automatic_remap import (
        bind_remap_execution,
        prepare_unstructured_conservative_remap,
    )
    from phydrax.meshing._measurements import NativeExecutionRecord
    from phydrax.meshing._volume_generation import native_volume_source_execution_budget

    adaptation, previous, old, new = native_metric_fv_owners
    density = jnp.full((old.cell_volumes.shape[0], 1), 2.0)
    history = jnp.arange(old.cell_volumes.shape[0], dtype=jnp.float64)[:, None] + 0.5
    before = np.asarray(history).copy()
    assert previous.owner_id is not None
    with (
        NativeExecutionScope(adaptation.policy.limits.native_limits()) as native,
        native_volume_source_execution_budget(
            adaptation.policy.limits, previous, previous.owner_id, scope=native
        ),
        native.host_workspace() as workspace,
    ):
        assert workspace is not None
        workspace.retain_owner((old, new, density, history))
        ticket = native.batch(0)
        ticket.settle()
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=adaptation.policy.limits.maximum_work_units,
            maximum_memory_bytes=adaptation.policy.limits.maximum_scratch_bytes,
        )
        with ledger.retained_storage(lambda size: native.batch(0, bytes_upper=size)):
            assert workspace is not ledger.storage_ticket
            owner_bound = workspace.bound
            prepared = prepare_unstructured_conservative_remap(
                old,
                new,
                provenance="real-native-resource-owner-content",
                adaptation=adaptation,
                source_workspace=workspace,
                scope=native,
                budget=ledger,
            )
            assert prepared.execution_evidence is None and prepared.plan is not None
            target_density = prepared.plan.apply(density)
            target_history = prepared.plan.apply(history)
            workspace.retain_owner((target_density, target_history))
            assert workspace.bound >= owner_bound
            ledger.reserve(0)
            assert workspace.bound >= owner_bound
            for source_value, target_value in (
                (density, target_density),
                (history, target_history),
            ):
                defect = prepared.plan.conservation_defect(source_value, target_value)
                assert bool(jnp.all(jnp.abs(defect) <= prepared.plan.report.tolerance))
    assert native.evidence is not None
    record = NativeExecutionRecord(
        native.evidence, owner_id=previous.owner_id, preparation_evidence=previous
    )
    finalized = bind_remap_execution(prepared, record)
    assert finalized.execution_evidence is not None
    finalized.execution_evidence.require_valid()
    assert finalized.execution_evidence.owner_id == record.owner_id
    assert (
        int(np.asarray(record.total_work_units))
        <= adaptation.policy.limits.maximum_work_units
    )
    assert (
        int(np.asarray(record.total_geometry_queries))
        <= adaptation.policy.limits.maximum_geometry_queries
    )
    np.testing.assert_array_equal(history, before)


def test_native_remap_rejects_algebra_scratch_as_persistent_owner(
    native_metric_fv_owners: _NativeMetricFVCase,
) -> None:
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.discretization.finite_volume._automatic_remap import (
        prepare_unstructured_conservative_remap,
    )
    from phydrax.meshing._volume_generation import native_volume_source_execution_budget

    adaptation, previous, old, new = native_metric_fv_owners
    before = np.asarray(old.mesh.coordinates).copy()
    assert previous.owner_id is not None
    with (
        NativeExecutionScope(adaptation.policy.limits.native_limits()) as native,
        native_volume_source_execution_budget(
            adaptation.policy.limits, previous, previous.owner_id, scope=native
        ),
    ):
        ticket = native.batch(0)
        ticket.settle()
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=adaptation.policy.limits.maximum_work_units,
            maximum_memory_bytes=adaptation.policy.limits.maximum_scratch_bytes,
        )
        with (
            ledger.retained_storage(lambda size: native.batch(0, bytes_upper=size)),
            pytest.raises(
                TypeError,
                match=r"prepare_unstructured_conservative_remap argument 'source_workspace'",
            ),
        ):
            prepare_unstructured_conservative_remap(
                old,
                new,
                provenance="invalid-scratch-owner",
                adaptation=adaptation,
                source_workspace=ledger.storage_ticket,  # ty: ignore[invalid-argument-type]
                scope=native,
                budget=ledger,
            )
    np.testing.assert_array_equal(old.mesh.coordinates, before)


def test_native_remap_unrelated_root_cannot_omit_actual_predecessor_cost(
    native_metric_fv_owners: _NativeMetricFVCase,
) -> None:
    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionLimits,
        NativeExecutionScope,
    )
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.discretization.finite_volume._automatic_remap import (
        prepare_unstructured_conservative_remap,
    )
    from phydrax.meshing import MeshingFailure

    adaptation, previous, old, new = native_metric_fv_owners
    priced_work = int(np.asarray(previous.total_work_units))
    assert priced_work > 0
    before = np.asarray(old.mesh.coordinates).copy()
    limits = adaptation.policy.limits
    native = NativeExecutionScope(
        NativeExecutionLimits(
            maximum_work_units=priced_work - 1,
            maximum_geometry_queries=limits.maximum_geometry_queries,
            maximum_cavity_cells=limits.maximum_cavity_cells,
            maximum_managed_bytes=limits.maximum_scratch_bytes,
            maximum_seconds=limits.maximum_wall_seconds,
        )
    )
    with pytest.raises((MeshingFailure, MeshcoreError)):
        with native, native.batch(0) as ticket:
            ticket.settle()
            ledger = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            prepare_unstructured_conservative_remap(
                old,
                new,
                provenance="unrelated-root-cannot-drop-source",
                adaptation=adaptation,
                scope=native,
                budget=ledger,
            )
    assert native.evidence is not None
    assert native.evidence.status == MeshcoreStatus.CAPACITY_EXCEEDED
    np.testing.assert_array_equal(old.mesh.coordinates, before)


def test_native_remap_invalid_provenance_remains_nonresource_exception(
    native_metric_fv_owners: _NativeMetricFVCase,
) -> None:
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.discretization.finite_volume._automatic_remap import (
        prepare_unstructured_conservative_remap,
    )
    from phydrax.meshing._volume_generation import native_volume_source_execution_budget

    adaptation, previous, old, new = native_metric_fv_owners
    before = np.asarray(old.cell_volumes).copy()
    assert previous.owner_id is not None
    with pytest.raises(ValueError, match="provenance"):
        with (
            NativeExecutionScope(adaptation.policy.limits.native_limits()) as native,
            native_volume_source_execution_budget(
                adaptation.policy.limits, previous, previous.owner_id, scope=native
            ),
            native.host_workspace() as workspace,
        ):
            assert workspace is not None
            ticket = native.batch(0)
            ticket.settle()
            ledger = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=adaptation.policy.limits.maximum_work_units,
                maximum_memory_bytes=adaptation.policy.limits.maximum_scratch_bytes,
            )
            with ledger.retained_storage(lambda size: native.batch(0, bytes_upper=size)):
                prepare_unstructured_conservative_remap(
                    old,
                    new,
                    provenance="",
                    adaptation=adaptation,
                    source_workspace=workspace,
                    scope=native,
                    budget=ledger,
                )
    np.testing.assert_array_equal(old.cell_volumes, before)


@pytest.fixture(scope="module")
def native_metric_geometry_receipt(
    native_metric_fv_owners: _NativeMetricFVCase,
) -> NativeExecutionRecord:
    """Append native coordinate predicates, preserving the source chain."""
    from phydrax._meshcore import exact_orient3d, NativeExecutionScope
    from phydrax.meshing._volume_generation import native_volume_source_execution_budget

    case = native_metric_fv_owners
    cells = np.asarray(case.source.mesh.blocks[0].vertices)
    tetrahedra = np.asarray(case.source.mesh.coordinates)[cells]
    assert case.previous.owner_id is not None
    with (
        NativeExecutionScope(case.adaptation.policy.limits.native_limits()) as native,
        native_volume_source_execution_budget(
            case.adaptation.policy.limits,
            case.previous,
            case.previous.owner_id,
            scope=native,
        ),
        native.host_workspace() as workspace,
    ):
        assert workspace is not None
        workspace.retain_owner((case.source, tetrahedra))
        sign = exact_orient3d(
            tetrahedra[:, 0], tetrahedra[:, 1], tetrahedra[:, 2], tetrahedra[:, 3]
        )
        determinant = np.linalg.det(tetrahedra[:, 1:] - tetrahedra[:, :1])
        np.testing.assert_array_equal(sign, np.sign(determinant))
    assert native.evidence is not None
    receipt = NativeExecutionRecord(
        native.evidence,
        owner_id=case.previous.owner_id,
        preparation_evidence=case.previous,
    )
    receipt.require_valid()
    assert int(np.asarray(receipt.total_geometry_queries)) >= len(cells)
    assert float(np.asarray(receipt.total_elapsed_seconds)) > 0.0
    return receipt


@pytest.mark.parametrize("resource", ["queries", "time"])
def test_native_consumer_admission_cannot_renew_actual_source_query_or_clock_cost(
    native_metric_fv_owners: _NativeMetricFVCase,
    native_metric_geometry_receipt: NativeExecutionRecord,
    resource: str,
) -> None:
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.meshing import MeshingFailure, MeshingLimits
    from phydrax.meshing._volume_generation import native_volume_source_execution_budget

    case = native_metric_fv_owners
    receipt = native_metric_geometry_receipt
    limits = case.adaptation.policy.limits
    starved = MeshingLimits(
        maximum_vertices=limits.maximum_vertices,
        maximum_edges=limits.maximum_edges,
        maximum_faces=limits.maximum_faces,
        maximum_cells=limits.maximum_cells,
        maximum_connectivity_entries=limits.maximum_connectivity_entries,
        maximum_data_bytes=limits.maximum_data_bytes,
        maximum_work_units=limits.maximum_work_units,
        maximum_cavity_cells=limits.maximum_cavity_cells,
        maximum_geometry_queries=(
            int(np.asarray(receipt.total_geometry_queries)) - 1
            if resource == "queries"
            else limits.maximum_geometry_queries
        ),
        maximum_scratch_bytes=limits.maximum_scratch_bytes,
        maximum_wall_seconds=(
            0.5 * float(np.asarray(receipt.total_elapsed_seconds))
            if resource == "time"
            else limits.maximum_wall_seconds
        ),
    )
    before = np.asarray(case.source.cell_volumes).copy()
    entered = False
    assert receipt.owner_id is not None
    with pytest.raises(MeshingFailure):
        with (
            NativeExecutionScope(starved.native_limits()) as native,
            native_volume_source_execution_budget(
                starved, receipt, receipt.owner_id, scope=native
            ),
            native.host_workspace() as workspace,
        ):
            entered = True
            assert workspace is not None
    assert not entered
    receipt.require_valid()
    np.testing.assert_array_equal(case.source.cell_volumes, before)


@pytest.fixture(scope="module")
def campaign_oracle_source() -> HPSource:
    return prepare_hp_source()


@pytest.fixture(scope="module")
def campaign_oracle_target(campaign_oracle_source: HPSource) -> HPTarget:
    return prepare_hp_target(campaign_oracle_source, AdaptationAction.P, 2)


def test_campaign_manufactured_source_and_observed_cost_have_independent_anchors() -> (
    None
):
    points = jnp.asarray(((0.0, 0.2), (0.5, 0.8), (1.0, 0.4)), dtype=jnp.float64)
    reference = np.asarray((0.0, 0.25, 1.0), dtype=np.float64)
    np.testing.assert_array_equal(exact(points), reference)
    np.testing.assert_array_equal(source(points, None), reference)
    assert peak_rss_bytes() > 0


def test_campaign_baseline_solve_matches_closed_reaction_projection(
    campaign_oracle_source: HPSource,
) -> None:
    case = campaign_oracle_source
    # Orthogonal projection of x² onto span{1, x}: x - 1/6.
    # Its squared error integrates to 1/180 and its integral is exactly 1/3.
    error, integral = error_and_integral(case.initial, case.values, 7)
    assert error == pytest.approx(np.sqrt(1.0 / 180.0), abs=_CAMPAIGN_CLOSED_MOMENT_ATOL)
    assert integral == pytest.approx(1.0 / 3.0, abs=_CAMPAIGN_CLOSED_MOMENT_ATOL)
    evidence = physical(case.initial, case.values, "closed-reaction-projection", 7)
    assert evidence.revision_id == case.initial.epoch_id
    assert evidence.bound == pytest.approx(error, abs=_CAMPAIGN_CLOSED_MOMENT_ATOL)


@pytest.mark.parametrize(
    ("value", "squared_error", "integral"),
    [(0.0, 1.0 / 5.0, 0.0), (1.0, 8.0 / 15.0, 1.0)],
    ids=["zero-field-fault", "constant-field-fault"],
)
def test_campaign_physical_oracle_detects_wrong_fields_by_closed_moments(
    campaign_oracle_target: HPTarget,
    value: float,
    squared_error: float,
    integral: float,
) -> None:
    epoch = campaign_oracle_target.epoch
    field = jnp.full(
        (epoch.discretization.field_spaces[0].layout.size,), value, dtype=jnp.float64
    )
    error, actual_integral = error_and_integral(epoch, field, 7)
    assert error == pytest.approx(
        np.sqrt(squared_error), abs=_CAMPAIGN_CLOSED_MOMENT_ATOL
    )
    assert actual_integral == pytest.approx(integral, abs=_CAMPAIGN_CLOSED_MOMENT_ATOL)
    assert error > 0.025


def test_campaign_pack_unpack_and_refresh_retain_nonzero_field(
    campaign_oracle_source: HPSource,
) -> None:
    case = campaign_oracle_source
    packed = pack(case.initial, case.values, 4)
    np.testing.assert_array_equal(unpack(case.initial, packed), case.values)
    refreshed = SolvedFieldRefresh(packed)((jnp.zeros_like(packed),), None, None)
    np.testing.assert_array_equal(refreshed[0], packed)
    assert bool(jnp.any(refreshed[0] != 0))


def test_campaign_named_history_transport_matches_closed_integrals(
    campaign_oracle_source: HPSource,
    campaign_oracle_target: HPTarget,
) -> None:
    initial, target = campaign_oracle_source.initial, campaign_oracle_target
    transfer = phx.discretization.fem.finite_element_hp_transfer_plan(
        initial, target.epoch, target.lineage, "u", target.kind
    )
    packed = pack(initial, campaign_oracle_source.values, transfer.primal.shape[2])
    auxiliary, integrator = HPHistoryTransfer(transfer)(
        (("previous-reaction-state", packed * 0.5),),
        (("accumulated-reaction-state", packed * 2.0),),
        None,
        None,
    )
    assert auxiliary[0][0] == "previous-reaction-state"
    assert integrator[0][0] == "accumulated-reaction-state"
    assert error_and_integral(target.epoch, unpack(target.epoch, auxiliary[0][1]), 7)[
        1
    ] == pytest.approx(1.0 / 6.0, abs=_CAMPAIGN_CLOSED_MOMENT_ATOL)
    assert error_and_integral(target.epoch, unpack(target.epoch, integrator[0][1]), 7)[
        1
    ] == pytest.approx(2.0 / 3.0, abs=_CAMPAIGN_CLOSED_MOMENT_ATOL)


def test_campaign_publication_certifiers_reject_zero_field_fault(
    campaign_oracle_target: HPTarget,
) -> None:
    epoch = campaign_oracle_target.epoch
    values = epoch.discretization.project("u", source)
    packed = pack(epoch, values, 9)
    hp = HPPhysicalCertifier(packed, jnp.asarray(1.0 / 3.0), 0.025)
    metric = MetricPhysicalCertifier(epoch.discretization, jnp.asarray(1.0 / 3.0), 0.025)
    assert hp(epoch, (packed,), None, None, None)
    assert metric(epoch.mesh, (values,), None, None, None)
    assert not hp(epoch, (jnp.zeros_like(packed),), None, None, None)
    assert not metric(epoch.mesh, (jnp.zeros_like(values),), None, None, None)
    reanalysis = MetricPhysicalReanalysis(
        epoch.discretization, epoch.epoch_id, "reaction-L2-error"
    )
    fault = reanalysis(epoch.mesh, (jnp.zeros_like(values),), None, None)
    assert fault.bound == pytest.approx(
        np.sqrt(1.0 / 5.0), abs=_CAMPAIGN_CLOSED_MOMENT_ATOL
    )
    assert fault.bound > 0.025


@pytest.mark.parametrize(
    ("helper", "field"),
    [
        ("exact", "accepted_L2_error"),
        ("source", "accepted_L2_error"),
        ("solve_problem", "accepted_L2_error"),
        ("compile_and_solve", "accepted_L2_error"),
        ("error_and_integral", "transfer_integral_error"),
        ("pack", "transfer_integral_error"),
        ("unpack", "transfer_integral_error"),
        ("physical", "accepted_L2_error"),
        ("SolvedFieldRefresh", "accepted_L2_error"),
        ("HPHistoryTransfer", "history_integral_errors"),
        ("HPPhysicalCertifier", "accepted_L2_error"),
        ("hp_transfer_derivatives", "transfer_derivatives"),
        ("prepare_hp_source", "accepted_L2_error"),
        ("prepare_hp_target", "actions"),
        ("stage_hp_trial", "history_integral_errors"),
        ("stage_hp_trial.capture", "history_integral_errors"),
        ("price_hp_trial", "phase_seconds"),
        ("publish_hp_trial", "accepted_L2_error"),
        ("hp_history_errors", "history_integral_errors"),
        ("run", "accepted_L2_error"),
        ("hp_campaign", "accepted_L2_error"),
    ],
    ids=[
        "exact",
        "source",
        "solve",
        "compile",
        "physical-integral",
        "pack",
        "unpack",
        "physical-evidence",
        "field-refresh",
        "hp-history",
        "hp-certifier",
        "hp-derivatives",
        "hp-source",
        "hp-target",
        "hp-staging",
        "hp-rebind",
        "hp-pricing",
        "hp-publication",
        "hp-history-errors",
        "hp-run",
        "hp-child",
    ],
)
def test_hp_campaign_oracle_rejects_corrupted_helper_output(
    hp_campaign_reports: dict[int, dict[str, Any]],
    helper: str,
    field: str,
) -> None:
    del helper
    published = hp_campaign_reports[2]
    _assert_hp_campaign(published)
    faulty = deepcopy(published)
    match field:
        case "accepted_L2_error" | "transfer_integral_error":
            faulty[field] = 1.0
        case "history_integral_errors":
            faulty[field]["previous-reaction-state"] = 1.0
        case "phase_seconds":
            faulty["trials"][0]["phase_seconds"] = [0.0] * 6
        case "actions":
            faulty["trials"][0]["action"] = "metric"
        case "transfer_derivatives":
            faulty["trials"][0][field]["JVP_error"] = 1.0
        case _:
            raise AssertionError(
                "campaign_fault.field must select a declared output invariant; got an unknown field"
            )
    with pytest.raises(AssertionError):
        _assert_hp_campaign(faulty)


@pytest.mark.parametrize(
    ("helper", "field"),
    [
        ("MetricPhysicalCertifier", "accepted_L2_error"),
        ("MetricPhysicalReanalysis", "accepted_L2_error"),
        ("prepare_metric_source", "accepted_L2_error"),
        ("stage_metric_trial", "integral_error"),
        ("stage_metric_trial.capture", "integral_error"),
        ("price_metric_trial", "accepted_L2_error"),
        ("publish_metric_trial", "accepted_L2_error"),
        ("refuse_metric_reanalysis", "physical_reanalysis_rollback"),
        ("refuse_metric_reanalysis.reject", "physical_reanalysis_rollback"),
        ("metric_transfer_derivatives", "transfer_JVP_finite_difference_error"),
        ("run_metric", "accepted_L2_error"),
    ],
    ids=[
        "metric-certifier",
        "metric-reanalysis",
        "metric-source",
        "metric-staging",
        "metric-rebind",
        "metric-pricing",
        "metric-publication",
        "metric-refusal",
        "metric-refusal-callback",
        "metric-derivatives",
        "metric-run",
    ],
)
def test_metric_campaign_oracle_rejects_corrupted_helper_output(
    metric_campaign_report: dict[str, Any],
    helper: str,
    field: str,
) -> None:
    del helper
    _assert_metric_campaign(metric_campaign_report)
    faulty = dict(metric_campaign_report)
    faulty[field] = False if field == "physical_reanalysis_rollback" else 1.0
    with pytest.raises(AssertionError):
        _assert_metric_campaign(faulty)


def test_hp_resume_oracle_rejects_corrupted_helper_output(
    archived_hp_campaign: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    authored, restored = archived_hp_campaign
    _assert_restored_hp_campaign(authored, restored)
    faulty = dict(restored, independently_recomputed_L2_error=1.0)
    with pytest.raises(AssertionError):
        _assert_restored_hp_campaign(authored, faulty)


def _assert_observed_peak_bytes(value: int) -> None:
    assert value > 0


def test_campaign_peak_rss_oracle_rejects_corrupted_helper_output() -> None:
    _assert_observed_peak_bytes(peak_rss_bytes())
    with pytest.raises(AssertionError):
        _assert_observed_peak_bytes(0)
