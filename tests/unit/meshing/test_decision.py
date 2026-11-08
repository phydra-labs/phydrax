from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax.linalg as la
from phydrax.geometry.design import (
    DerivativeTier,
    DesignQualificationEvidence,
    ParameterId,
)
from phydrax.meshing._decision import (
    AdaptationAction,
    ErrorQuantity,
    FixedEpochDerivativeEvidence,
    MeasuredAdaptationCost,
    MeshAdaptationCandidate,
    MeshAdaptationObjective,
    PhysicalErrorEvidence,
    RouteFeasibility,
)


def _solve_check(
    *, finite: bool = True, operator_id: str = "decision-solve-check"
) -> la.LinearSolveCheckEvidence:
    operator = la.DenseLinearOperator(
        jnp.eye(2, dtype=jnp.float64), operator_id=operator_id
    )
    problem = la.LinearSystem(operator)
    rhs = (
        jnp.asarray((1.0, 2.0), dtype=jnp.float64)
        if finite
        else jnp.asarray((jnp.inf, 2.0), dtype=jnp.float64)
    )
    _, evidence = la.solve_checked(
        problem,
        rhs,
        policy=la.LinearSolvePolicy(la.DenseLU()),
    )
    return evidence


def _error(
    revision: str,
    estimator: str,
    /,
    *,
    field: float,
    geometry: float,
    qoi_certificate_id: str | None = None,
    quantity: ErrorQuantity = "physical-error",
) -> PhysicalErrorEvidence:
    return PhysicalErrorEvidence(
        revision,
        "sphere-x2-moment" if quantity == "qoi" else "sphere-field-error",
        estimator,
        field_error=field,
        geometry_error=geometry,
        algebraic_error=0.0,
        transfer_error=0.0,
        quantity=quantity,
        qoi_certificate_id=qoi_certificate_id,
    )


def _candidate(
    candidate_id: str,
    action: AdaptationAction,
    /,
    *,
    field: float,
    geometry: float,
    seconds: float,
    solve_check: la.LinearSolveCheckEvidence,
    issues: tuple[str, ...] = (),
    quantity: ErrorQuantity = "physical-error",
    qoi_certificate_id: str | None = None,
    required_state_ids: tuple[str, ...] = (),
) -> MeshAdaptationCandidate:
    target = f"target:{candidate_id}"
    feasibility = RouteFeasibility(
        "accepted",
        target,
        f"route:{candidate_id}",
        candidate_id,
        cell_families=("triangle",),
        geometry_layout_id=f"geometry:{candidate_id}",
        field_layouts=(("u", f"field-layout:{candidate_id}"),),
        solver_operator_id="decision-solve-check",
        transfer_plan_id=f"transfer:{candidate_id}",
        compiled_layout_id=f"compiled:{candidate_id}",
        geometry_certificate_id=f"geometry-certificate:{candidate_id}",
        topology_certificate_id=f"topology-certificate:{candidate_id}",
        required_state_ids=required_state_ids,
        state_dispositions=tuple(
            (identifier, f"disposition:{identifier}") for identifier in required_state_ids
        ),
        issues=issues,
    )
    return MeshAdaptationCandidate(
        action,
        candidate_id,
        feasibility,
        _error(
            target,
            f"estimator:{candidate_id}",
            field=field,
            geometry=geometry,
            quantity=quantity,
            qoi_certificate_id=qoi_certificate_id,
        ),
        MeasuredAdaptationCost(
            candidate_id,
            preparation_seconds=0.1,
            compilation_seconds=0.0,
            adaptation_seconds=seconds,
            transfer_seconds=0.1,
            reanalysis_seconds=0.2,
        ),
        solve_check,
    )


def test_objective_ranks_measured_cost_only_after_physical_and_numeric_gates() -> None:
    objective = MeshAdaptationObjective("sphere-field-error", 0.1)
    baseline = _error("accepted", "baseline-estimator", field=0.5, geometry=0.1)
    valid = _solve_check()
    invalid = _solve_check(finite=False)
    cheap_but_numerically_invalid = _candidate(
        "cheap-invalid",
        AdaptationAction.METRIC,
        field=0.01,
        geometry=0.01,
        seconds=0.01,
        solve_check=invalid,
    )
    expensive_but_route_invalid = _candidate(
        "route-invalid",
        AdaptationAction.H,
        field=0.01,
        geometry=0.01,
        seconds=0.2,
        solve_check=valid,
        issues=("transfer-unavailable",),
    )
    feasible = _candidate(
        "feasible",
        AdaptationAction.P,
        field=0.03,
        geometry=0.04,
        seconds=0.1,
        solve_check=valid,
    )
    decision = objective.decide(
        baseline,
        (expensive_but_route_invalid, feasible, cheap_but_numerically_invalid),
    )

    assert decision.selected is feasible
    assert dict(decision.dispositions)["cheap-invalid"] == (
        "numerical-solve-unadmissible",
    )
    assert dict(decision.dispositions)["route-invalid"] == ("transfer-unavailable",)


@pytest.mark.parametrize("action", tuple(AdaptationAction))
def test_each_adaptation_action_is_ranked_by_the_same_physical_objective(
    action: AdaptationAction,
) -> None:
    objective = MeshAdaptationObjective("sphere-field-error", 0.1)
    baseline = _error("accepted", "baseline", field=0.5, geometry=0.1)
    candidate_id = f"action-{action.value}"
    candidate = _candidate(
        candidate_id,
        action,
        field=0.02,
        geometry=0.02,
        seconds=0.1,
        solve_check=_solve_check(),
    )

    decision = objective.decide(baseline, (candidate,))

    assert decision.selected is candidate


def test_objective_stops_without_remeshing_when_baseline_meets_tolerance() -> None:
    objective = MeshAdaptationObjective("sphere-field-error", 0.1)
    baseline = _error("accepted", "baseline", field=0.03, geometry=0.01)
    candidate = _candidate(
        "unneeded-remesh",
        AdaptationAction.H,
        field=0.005,
        geometry=0.005,
        seconds=0.1,
        solve_check=_solve_check(),
    )

    decision = objective.decide(baseline, (candidate,))

    assert decision.selected is None
    assert decision.satisfied_without_adaptation
    assert dict(decision.dispositions)["unneeded-remesh"] == (
        "baseline-already-meets-target",
    )


def test_explicit_improve_intent_preserves_target_and_admissibility_gates() -> None:
    objective = MeshAdaptationObjective("sphere-field-error", 0.1, intent="improve")
    default = MeshAdaptationObjective("sphere-field-error", 0.1)
    assert objective.objective_id_hash != default.objective_id_hash
    baseline = _error("accepted", "baseline", field=0.03, geometry=0.01)
    check = _solve_check()
    good = _candidate(
        "good",
        AdaptationAction.H,
        field=0.005,
        geometry=0.005,
        seconds=1.0,
        solve_check=check,
    )
    unchanged = _candidate(
        "unchanged",
        AdaptationAction.H,
        field=0.03,
        geometry=0.01,
        seconds=0.0,
        solve_check=check,
    )
    invalid = _candidate(
        "invalid",
        AdaptationAction.H,
        field=0.001,
        geometry=0.001,
        seconds=0.0,
        solve_check=check,
        issues=("transfer-unavailable",),
    )
    unmet = _candidate(
        "unmet",
        AdaptationAction.H,
        field=0.2,
        geometry=0.0,
        seconds=0.0,
        solve_check=check,
    )
    decision = objective.decide(baseline, (good, unchanged, invalid, unmet))
    assert decision.selected is good
    assert not decision.satisfied_without_adaptation
    dispositions = dict(decision.dispositions)
    assert "physical-error-not-improved" in dispositions["unchanged"]
    assert "transfer-unavailable" in dispositions["invalid"]
    assert "physical-target-unmet" in dispositions["unmet"]
    stale = objective.decide(
        _error("another-source", "baseline", field=0.03, geometry=0.01), (good,)
    )
    assert stale.selected is None
    assert "stale-source" in dict(stale.dispositions)["good"]
    assert not stale.satisfied_without_adaptation


def test_foreign_solver_evidence_cannot_admit_a_route() -> None:
    objective = MeshAdaptationObjective("sphere-field-error", 0.1)
    baseline = _error("accepted", "baseline", field=0.5, geometry=0.1)
    candidate = _candidate(
        "foreign-operator",
        AdaptationAction.METRIC,
        field=0.01,
        geometry=0.01,
        seconds=0.01,
        solve_check=_solve_check(operator_id="foreign-operator"),
    )

    decision = objective.decide(baseline, (candidate,))

    assert decision.selected is None
    assert dict(decision.dispositions)["foreign-operator"] == ("solver-binding-mismatch",)


def test_physical_error_components_cannot_cancel_between_geometry_and_field() -> None:
    objective = MeshAdaptationObjective("sphere-field-error", 0.1)
    baseline = _error("accepted", "baseline-estimator", field=0.5, geometry=0.2)
    candidate = _candidate(
        "geometry-dominated",
        AdaptationAction.GEOMETRY_ORDER,
        field=0.01,
        geometry=0.11,
        seconds=0.1,
        solve_check=_solve_check(),
    )
    decision = objective.decide(baseline, (candidate,))

    assert decision.selected is None
    assert dict(decision.dispositions)["geometry-dominated"] == ("physical-target-unmet",)
    assert candidate.error.bound == pytest.approx(0.12)


def test_qoi_objective_requires_a_well_posed_target_and_independent_reanalysis() -> None:
    with pytest.raises(ValueError, match="well-posedness certificate"):
        PhysicalErrorEvidence(
            "accepted",
            "sphere-x2-moment",
            "dwr-estimator",
            field_error=0.2,
            geometry_error=0.1,
            algebraic_error=0.0,
            transfer_error=0.0,
            quantity="qoi",
        )

    objective = MeshAdaptationObjective("sphere-x2-moment", 0.1, quantity="qoi")
    baseline = _error(
        "accepted",
        "baseline-dwr",
        field=0.3,
        geometry=0.1,
        quantity="qoi",
        qoi_certificate_id="baseline-qoi-certificate",
    )
    candidate = _candidate(
        "combined-design-target",
        AdaptationAction.METRIC,
        field=0.03,
        geometry=0.02,
        seconds=0.2,
        solve_check=_solve_check(),
        quantity="qoi",
        qoi_certificate_id="target-qoi-certificate",
    )
    decision = objective.decide(baseline, (candidate,))
    actual = _error(
        candidate.feasibility.target_revision_id,
        "independent-continuous-qoi-oracle",
        field=0.04,
        geometry=0.01,
        quantity="qoi",
        qoi_certificate_id="target-qoi-certificate",
    )

    assert decision.require_reanalysis(actual, _solve_check()) is candidate

    same_estimator = _error(
        candidate.feasibility.target_revision_id,
        candidate.error.estimator_id,
        field=0.04,
        geometry=0.01,
        quantity="qoi",
        qoi_certificate_id="target-qoi-certificate",
    )
    assert decision.reanalysis_issues(same_estimator, _solve_check()) == (
        "independent-reanalysis-required",
    )


def test_reanalysis_requires_the_selected_target_and_qoi_certificate() -> None:
    objective = MeshAdaptationObjective("sphere-x2-moment", 0.1, quantity="qoi")
    baseline = _error(
        "accepted",
        "baseline",
        field=0.3,
        geometry=0.1,
        quantity="qoi",
        qoi_certificate_id="baseline-certificate",
    )
    candidate = _candidate(
        "selected",
        AdaptationAction.H,
        field=0.04,
        geometry=0.02,
        seconds=0.1,
        solve_check=_solve_check(),
        quantity="qoi",
        qoi_certificate_id="target-certificate",
    )
    decision = objective.decide(baseline, (candidate,))
    foreign = _error(
        "foreign-target",
        "independent-oracle",
        field=0.04,
        geometry=0.02,
        quantity="qoi",
        qoi_certificate_id="foreign-certificate",
    )

    assert decision.reanalysis_issues(foreign, _solve_check()) == (
        "reanalysis-qoi-certificate-mismatch",
        "reanalysis-target-mismatch",
    )


def test_required_state_rebind_cannot_be_skipped_before_reanalysis() -> None:
    objective = MeshAdaptationObjective("sphere-field-error", 0.1)
    baseline = _error("accepted", "baseline", field=0.4, geometry=0.1)
    candidate = _candidate(
        "field-transfer",
        AdaptationAction.H,
        field=0.03,
        geometry=0.02,
        seconds=0.1,
        solve_check=_solve_check(),
        required_state_ids=("field/u", "history/plastic"),
    )
    decision = objective.decide(baseline, (candidate,))
    actual = _error(
        candidate.feasibility.target_revision_id,
        "independent-reanalysis",
        field=0.04,
        geometry=0.01,
    )

    assert decision.reanalysis_issues(actual, _solve_check()) == (
        "composition-rebind-required",
    )


def _qualification(
    epoch: str, plan: str, event: str, /, *, margin: float = 0.5
) -> DesignQualificationEvidence:
    return DesignQualificationEvidence(
        DerivativeTier.Q1,
        plan,
        epoch,
        plan,
        gradient_error=1.0e-12,
        event_ids=(event,),
        event_margins=(margin,),
        valid=True,
    )


def _fixed_derivatives() -> FixedEpochDerivativeEvidence:
    epoch = "sphere-accepted-epoch"
    plans = ("sphere-geometry", "sphere-pde", "sphere-transfer")
    events = ("geometry-singularity", "pde-coercivity", "donor-switch")
    qualifications = tuple(
        _qualification(epoch, plan, event)
        for plan, event in zip(plans, events, strict=True)
    )
    return FixedEpochDerivativeEvidence(
        epoch,
        ParameterId("design-sphere", "radius"),
        geometry=qualifications[0],
        pde=qualifications[1],
        transfer=qualifications[2],
        routes=tuple(zip(("geometry", "pde", "transfer"), plans, strict=True)),
    )


def test_fixed_epoch_derivatives_reject_event_boundaries_and_discrete_edits() -> None:
    evidence = _fixed_derivatives()
    parameter = ParameterId("design-sphere", "radius")
    routes = evidence.routes
    margins = (
        ("geometry-singularity", 0.5),
        ("pde-coercivity", 0.5),
        ("donor-switch", 0.5),
    )
    evidence.require_valid(
        evidence.epoch_id,
        parameter,
        routes=routes,
        event_margins=margins,
    )

    assert "event-boundary" in evidence.invalidation_issues(
        evidence.epoch_id,
        parameter,
        routes=routes,
        event_margins=(
            ("geometry-singularity", 0.5),
            ("pde-coercivity", 0.5),
            ("donor-switch", 0.0),
        ),
    )
    assert "stopped-topology-event" in evidence.invalidation_issues(
        evidence.epoch_id,
        parameter,
        routes=routes,
        event_margins=margins,
        topology_event_id="h-refinement",
    )
    assert "stopped-geometry-realization-event" in evidence.invalidation_issues(
        evidence.epoch_id,
        parameter,
        routes=routes,
        event_margins=margins,
        geometry_realization_id="curved-map-changed",
    )
    assert "epoch-or-parameter-changed" in evidence.invalidation_issues(
        evidence.epoch_id,
        ParameterId("design-sphere", "center"),
        routes=routes,
        event_margins=margins,
    )
