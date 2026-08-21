#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _fold_problem():
    return phx.continuation.ParameterContinuationProblem(
        lambda state, parameter, _: {
            "x": state["x"] ** 2 + parameter - 1.0,
        },
        problem_id="quadratic-fold",
    )


def test_public_prepare_refresh_run_contract_preserves_symbolic_identity():
    problem = _fold_problem()
    plan = phx.continuation.plan_continuation(
        problem,
        num_steps=2,
        method=phx.continuation.PseudoArclengthContinuation(initial_step=0.1),
        plan_id="fold-plan",
    )
    prepared = phx.continuation.prepare_continuation(
        problem,
        {"x": jnp.asarray(1.0)},
        jnp.asarray(0.0),
        plan,
    )
    refreshed = phx.continuation.refresh_continuation(
        prepared,
        {"x": jnp.asarray(0.9)},
        jnp.asarray(0.19),
    )

    assert prepared.plan.plan_id == "fold-plan"
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id == prepared.prepared_id
    assert int(prepared.numeric_version) == 0
    assert int(refreshed.numeric_version) == 1
    result = phx.continuation.run_continuation(prepared)
    assert result.provenance.plan_id == "fold-plan"
    assert result.provenance.prepared_id == prepared.prepared_id
    assert result.provenance.corrector_id == "newton/gmres"
    assert int(result.provenance.numeric_version) == 0
    assert result.provenance.linear_reuse_mode == "prepared-newton"
    assert int(result.diagnostics.corrector_linear_preparations) == 1
    assert int(result.diagnostics.corrector_linear_numeric_refreshes) >= 0
    assert int(result.diagnostics.corrector_linear_solves) >= 1
    assert result.provenance.corrector_linear_plan_id
    assert int(result.provenance.corrector_linear_numeric_version) >= 0
    assert result.provenance.corrector_preconditioner_plan_id == ""


def test_preconditioned_corrector_reports_preserved_linear_plan_identity():
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, _: state - jnp.asarray([coordinate, 2.0 * coordinate]),
        problem_id="preconditioned-linear-curve",
    )
    linear_policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(),
        preconditioning=phx.linalg.PreconditioningPolicy(
            phx.linalg.JacobiPreconditionerBuilder()
        ),
    )

    result = phx.continuation.continue_branch(
        problem,
        jnp.zeros((2,)),
        jnp.asarray(0.0),
        num_steps=2,
        method=phx.continuation.NaturalParameterContinuation(
            linear_policy=linear_policy,
            initial_step=0.1,
        ),
    )

    assert result.status == phx.continuation.ContinuationStatus.SUCCESS
    assert result.provenance.corrector_linear_plan_id
    assert result.provenance.corrector_preconditioner_plan_id
    assert int(result.diagnostics.corrector_linear_preparations) == 1
    assert int(result.diagnostics.corrector_linear_numeric_refreshes) >= 1
    assert int(result.provenance.corrector_linear_numeric_version) >= 1


def test_pseudo_arclength_traverses_fold_with_uncertified_explicit_bracket():
    result = phx.continuation.continue_branch(
        _fold_problem(),
        {"x": jnp.asarray(1.0)},
        jnp.asarray(0.0),
        num_steps=12,
        method=phx.continuation.PseudoArclengthContinuation(
            initial_step=0.18,
            maximum_step=0.24,
            residual_tolerance=1e-8,
        ),
    )

    assert result.status == phx.continuation.ContinuationStatus.SUCCESS
    assert len(result.points) == 13
    assert result.fold_brackets
    bracket = result.fold_brackets[0]
    assert bracket.kind == "fold-candidate"
    assert not bracket.certified
    assert float(bracket.left_indicator) * float(bracket.right_indicator) <= 0.0
    parameter_tangents = np.asarray(
        [float(point.tangent_coordinate) for point in result.points]
    )
    assert np.any(parameter_tangents > 0.0)
    assert np.any(parameter_tangents < 0.0)
    assert max(float(point.residual_norm) for point in result.points) <= 1e-8
    assert any(event.kind == "fold-candidate" for event in result.events)


def test_event_localization_refines_fold_indicator_without_certifying_it():
    problem = _fold_problem()
    result = phx.continuation.continue_branch(
        problem,
        {"x": jnp.asarray(1.0)},
        jnp.asarray(0.0),
        num_steps=12,
        method=phx.continuation.PseudoArclengthContinuation(
            initial_step=0.18,
            maximum_step=0.24,
            residual_tolerance=1e-8,
        ),
    )
    bracket = result.fold_brackets[0]

    localized = phx.continuation.localize_event(
        problem,
        result.branch,
        bracket,
        lambda problem, state, coordinate, args: state["x"],
        indicator_id="quadratic-fold/state-zero",
        policy=phx.continuation.EventLocalizationPolicy(
            bracket_tolerance=1e-6,
            indicator_tolerance=1e-6,
            residual_tolerance=1e-8,
            maximum_steps=16,
        ),
    )

    assert int(localized.status) == int(phx.continuation.EventLocalizationStatus.SUCCESS)
    assert localized.point is not None
    assert abs(float(localized.point.state["x"])) <= 1e-6
    assert abs(float(localized.point.coordinate) - 1.0) <= 1e-6
    assert float(localized.point.residual_norm) <= 1e-8
    assert int(localized.diagnostics.bordered_preparations) == 1
    assert int(localized.diagnostics.bordered_numeric_refreshes) >= 0
    assert int(localized.diagnostics.cached_bordered_column_reuses) == (
        int(localized.diagnostics.bordered_preparations)
        + int(localized.diagnostics.bordered_numeric_refreshes)
    )
    assert int(localized.provenance.bordered_numeric_version) == int(
        localized.diagnostics.bordered_numeric_refreshes
    )
    assert localized.provenance.bordered_plan_id.endswith(
        "/event-localization-bordered-plan"
    )
    assert localized.provenance.bracket_id == bracket.bracket_id
    assert localized.provenance.indicator_id == "quadratic-fold/state-zero"
    assert not bracket.certified

    invalid = phx.continuation.localize_event(
        problem,
        result.branch,
        bracket,
        lambda problem, state, coordinate, args: jnp.asarray(1.0),
        indicator_id="constant-positive",
    )
    assert int(invalid.status) == int(
        phx.continuation.EventLocalizationStatus.INVALID_BRACKET
    )
    assert invalid.point is None


def test_natural_parameter_continuation_exposes_turning_point_limitation():
    result = phx.continuation.continue_branch(
        _fold_problem(),
        {"x": jnp.asarray(1.0)},
        jnp.asarray(0.0),
        num_steps=20,
        method=phx.continuation.NaturalParameterContinuation(
            initial_step=0.2,
            minimum_step=0.02,
            maximum_step=0.2,
            contraction=0.5,
            maximum_retries=2,
            residual_tolerance=1e-9,
        ),
    )

    assert result.status == phx.continuation.ContinuationStatus.CORRECTOR_FAILED
    assert result.termination_reason == "corrector recovery exhausted"
    assert not result.fold_brackets
    assert all(float(point.coordinate) <= 1.0 for point in result.points)
    assert all(float(point.tangent_coordinate) >= 0.0 for point in result.points)
    assert any(event.kind == "corrector-failure" for event in result.events)


def test_hopf_monitoring_records_conjugate_pair_crossing_bracket():
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, parameter, _: (
            jnp.asarray([[parameter, -1.0], [1.0, parameter]]) @ state
        ),
        parameter_lower=-1.0,
        parameter_upper=1.0,
        problem_id="complex-pair-crossing",
    )
    result = phx.continuation.continue_branch(
        problem,
        jnp.zeros((2,)),
        jnp.asarray(-0.25),
        num_steps=6,
        method=phx.continuation.PseudoArclengthContinuation(
            initial_step=0.11,
            maximum_step=0.11,
            residual_tolerance=1e-9,
        ),
        initial_tangent=(jnp.zeros((2,)), jnp.asarray(1.0)),
        stability_analyzer=phx.continuation.DenseSchurStabilityAnalyzer(),
    )

    assert result.status == phx.continuation.ContinuationStatus.SUCCESS
    assert len(result.branch.stability_points) == len(result.points)
    assert result.hopf_brackets
    bracket = result.hopf_brackets[0]
    assert bracket.kind == "hopf-candidate"
    assert not bracket.certified
    assert float(bracket.left_indicator) < 0.0
    assert float(bracket.right_indicator) >= 0.0
    event = next(event for event in result.events if event.kind == "hopf-candidate")
    assert event.bracket_id == bracket.bracket_id
    assert "not a certified bifurcation" in event.message


def test_general_krylov_stability_uses_public_restarted_arnoldi_contract():
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, _: (
            jnp.asarray(
                [
                    [coordinate, -1.0, 0.0, 0.0],
                    [1.0, coordinate, 0.0, 0.0],
                    [0.0, 0.0, -2.0, 0.0],
                    [0.0, 0.0, 0.0, -3.0],
                ]
            )
            @ state
        ),
        problem_id="general-krylov-complex-pair",
    )
    analyzer = phx.continuation.GeneralKrylovStabilityAnalyzer(
        mode_count=2,
        zero_tolerance=1e-8,
        pair_tolerance=1e-6,
    )

    evidence = analyzer.analyze(
        problem,
        jnp.zeros((4,)),
        jnp.asarray(0.2),
    )

    assert bool(evidence.successful)
    assert not evidence.full_spectrum
    assert int(evidence.conjugate_pair_count) == 1
    np.testing.assert_allclose(
        float(evidence.leading_complex_real_part),
        0.2,
        atol=1e-5,
    )
    assert evidence.analyzer_id == "general-krylov-stability"


def test_rejected_correctors_contract_step_deterministically():
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, parameter, _: state**2 + parameter + 1.0,
        problem_id="no-real-root",
    )
    result = phx.continuation.continue_branch(
        problem,
        jnp.asarray([0.0]),
        jnp.asarray(-1.0),
        num_steps=1,
        method=phx.continuation.NaturalParameterContinuation(
            initial_step=0.2,
            minimum_step=0.025,
            maximum_step=0.2,
            contraction=0.5,
            maximum_retries=3,
            maximum_corrector_steps=3,
            residual_tolerance=1e-12,
        ),
    )

    retries = [
        float(event.indicator)
        for event in result.events
        if event.kind == "corrector-retry"
    ]
    np.testing.assert_allclose(retries, np.asarray([0.1, 0.05, 0.025, 0.025]))
    assert all(
        int(event.source_status) != int(phx.linalg.LinearSolveStatus.SUCCESS)
        for event in result.events
        if event.kind == "corrector-retry"
    )
    assert result.status == phx.continuation.ContinuationStatus.CORRECTOR_FAILED
    assert int(result.diagnostics.rejected_steps) == 4
    assert int(result.diagnostics.attempted_steps) == 4


def test_branch_monitor_and_switch_hook_consume_explicit_branch_models():
    observed = []

    def monitor(problem, previous, current, args):
        del problem, previous, args
        observed.append(current.point_id)
        return (
            phx.continuation.ContinuationEvent(
                "user",
                current.coordinate,
                point_id=current.point_id,
                message="observed",
            ),
        )

    result = phx.continuation.continue_branch(
        _fold_problem(),
        {"x": jnp.asarray(1.0)},
        jnp.asarray(0.0),
        num_steps=2,
        method=phx.continuation.PseudoArclengthContinuation(initial_step=0.1),
        monitors=(phx.continuation.CallableBranchMonitor(monitor),),
    )

    assert observed == [point.point_id for point in result.points]
    assert int(result.diagnostics.monitor_events) == len(result.points)
    assert result.provenance.monitor_ids == ("callable-branch-monitor",)

    def propose(branch, event, args):
        del args
        if event.kind != "user":
            return ()
        point = next(point for point in branch.points if point.point_id == event.point_id)
        return (
            phx.continuation.BranchSeed(
                state=point.state,
                coordinate=point.coordinate,
                tangent_state=point.tangent_state,
                tangent_coordinate=point.tangent_coordinate,
                branch_id=f"switch-{point.point_id}",
                source_point_id=point.point_id,
            ),
        )

    seeds = phx.continuation.propose_branch_seeds(
        result.branch,
        (phx.continuation.CallableBranchSwitchHook(propose),),
    )
    assert len(seeds) == len(result.points)
