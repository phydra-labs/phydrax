from __future__ import annotations

from math import log, sqrt

import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_dynamics import (
    prepare_rod,
    RodPlan,
    RodState,
)
from phydrax.applications.solid_mechanics._rod_tendon import (
    FrictionlessElasticTendonPlan,
    RodMaterialStation,
    TendonActuatorState,
    TendonPayoutCommand,
    TendonRoutePlan,
)
from phydrax.applications.solid_mechanics._rod_tendon_friction import (
    CapstanTendonFrictionPlan,
    CapstanTendonFrictionState,
)
from phydrax.nonlinear import NonlinearTermination


def _rod_and_route():
    rod = prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
            jnp.ones((3,)),
            jnp.ones((2,)),
            jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
            jnp.ones((1, 1, 1)),
        )
    )
    route_plan = TendonRoutePlan(
        (
            RodMaterialStation(0, 0.0, jnp.zeros((2,))),
            RodMaterialStation(0, 1.0, jnp.zeros((2,))),
            RodMaterialStation(1, 1.0, jnp.zeros((2,))),
        )
    )
    return rod, route_plan, route_plan.prepare(rod)


def _state(tensions=(10.0, 30.0), slip=0.0):
    return CapstanTendonFrictionState(
        jnp.asarray(tensions),
        jnp.ones((2,)),
        jnp.asarray((slip,)),
    )


def test_zero_friction_recovers_the_ideal_series_tendon_tension():
    rod, route_plan, _ = _rod_and_route()
    ideal = FrictionlessElasticTendonPlan(
        route_plan,
        50.0,
        free_length_bounds=(0.5, 3.0),
        payout_rate_bounds=(-1.0, 1.0),
        tendon_length_bounds=(0.5, 4.0),
        maximum_tension=100.0,
    ).prepare(rod)
    capstan = CapstanTendonFrictionPlan(
        jnp.asarray((0.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray((100.0, 100.0)),
    ).prepare(ideal.route, _state())
    deformed_rod = RodState(
        jnp.asarray(((0.0, 0.0), (1.1, 0.0), (2.4, 0.0))),
        jnp.zeros((3, 2)),
        rod.rest_orientations,
        jnp.zeros((2,)),
    )
    _, span_lengths, _ = ideal.route.span_geometry(deformed_rod)
    span_length_rates = ideal.route.span_length_rates(deformed_rod)

    ideal_evaluation = ideal.evaluate(
        deformed_rod,
        TendonActuatorState(2.0),
        TendonPayoutCommand(0.0),
    )
    capstan_evaluation = capstan.evaluate(
        _state(),
        span_lengths,
        span_length_rates,
        0.1,
    )

    assert capstan_evaluation.successful
    assert ideal_evaluation.valid
    assert capstan_evaluation.candidate_state.tensions == pytest.approx(
        jnp.full((2,), ideal_evaluation.tension), rel=2.0e-5, abs=2.0e-5
    )
    assert capstan_evaluation.candidate_state.stress_free_lengths == pytest.approx(
        jnp.asarray((11.0 / 12.0, 13.0 / 12.0)), rel=2.0e-5, abs=2.0e-5
    )


def test_taut_tendon_sticks_strictly_inside_both_capstan_inequalities():
    _, _, route = _rod_and_route()
    state = _state((10.0, 15.0), slip=0.125)
    prepared = CapstanTendonFrictionPlan(
        jnp.asarray((log(2.0),)),
        jnp.asarray((1.0,)),
        jnp.asarray((100.0, 100.0)),
    ).prepare(route, state)

    evaluation = prepared.evaluate(
        state,
        jnp.asarray((1.1, 1.15)),
        jnp.asarray((0.2, -0.1)),
        0.05,
    )

    assert evaluation.successful
    assert evaluation.directional_slip_increment == pytest.approx(jnp.zeros((2, 1)))
    assert evaluation.candidate_state.stress_free_lengths == pytest.approx(
        state.stress_free_lengths
    )
    assert evaluation.candidate_state.slip == pytest.approx(state.slip)
    assert jnp.all(evaluation.forward_capstan_margin > 0.0)
    assert jnp.all(evaluation.reverse_capstan_margin > 0.0)
    assert evaluation.evidence.dissipation_power == pytest.approx(0.0, abs=1.0e-7)
    assert evaluation.evidence.power_residual == pytest.approx(0.0, abs=1.0e-6)


def test_taut_sliding_state_matches_capstan_boundary_and_dissipates():
    rod, _, route = _rod_and_route()
    state = _state()
    prepared = CapstanTendonFrictionPlan(
        jnp.asarray((log(2.0),)),
        jnp.asarray((1.0,)),
        jnp.asarray((100.0, 100.0)),
    ).prepare(route, state)
    expected_slip = 0.5 * (-3.5 + sqrt(12.65))
    deformed_rod = RodState(
        jnp.asarray(((0.0, 0.0), (1.1, 0.0), (2.4, 0.0))),
        jnp.asarray(((0.0, 0.0), (0.1, 0.0), (0.3, 0.0))),
        rod.rest_orientations,
        jnp.zeros((2,)),
    )
    _, span_lengths, _ = route.span_geometry(deformed_rod)
    span_length_rates = route.span_length_rates(deformed_rod)

    evaluation = prepared.evaluate(
        state,
        span_lengths,
        span_length_rates,
        0.1,
    )

    assert evaluation.successful
    assert evaluation.directional_slip_increment[0, 0] == pytest.approx(
        expected_slip, rel=3.0e-5, abs=3.0e-5
    )
    assert evaluation.directional_slip_increment[1, 0] == pytest.approx(0.0, abs=3.0e-6)
    assert evaluation.candidate_state.stress_free_lengths == pytest.approx(
        jnp.asarray((1.0 - expected_slip, 1.0 + expected_slip)),
        rel=3.0e-5,
        abs=3.0e-5,
    )
    assert evaluation.forward_capstan_margin[0] == pytest.approx(0.0, abs=2.0e-4)
    assert evaluation.candidate_state.tensions[1] == pytest.approx(
        2.0 * evaluation.candidate_state.tensions[0], abs=2.0e-4
    )
    assert evaluation.evidence.dissipation_nonnegative
    assert evaluation.evidence.interface_dissipation[0] > 0.0
    assert evaluation.evidence.dissipation_power > 0.0
    assert evaluation.evidence.power_residual == pytest.approx(0.0, abs=2.0e-5)
    assert evaluation.evidence.power_balanced
    span_effort = route.native_span_effort(
        deformed_rod, evaluation.candidate_state.tensions
    )
    paired_rod_power = rod.effort_space.pair(
        span_effort, rod.velocity_from_state(deformed_rod)
    ).real
    assert evaluation.evidence.rod_power == pytest.approx(paired_rod_power, abs=2.0e-5)
    assert jnp.array_equal(
        evaluation.accepted_state.tensions, evaluation.candidate_state.tensions
    )
    assert jnp.array_equal(
        evaluation.accepted_state.stress_free_lengths,
        evaluation.candidate_state.stress_free_lengths,
    )
    assert jnp.array_equal(
        evaluation.accepted_state.slip, evaluation.candidate_state.slip
    )


def test_fully_slack_spans_remain_finite_without_logarithmic_tensions():
    _, _, route = _rod_and_route()
    state = _state((0.0, 0.0), slip=-0.25)
    prepared = CapstanTendonFrictionPlan(
        jnp.asarray((0.8,)),
        jnp.asarray((2.0,)),
        jnp.asarray((100.0, 120.0)),
    ).prepare(route, state)

    evaluation = prepared.evaluate(
        state,
        jnp.asarray((0.8, 0.7)),
        jnp.asarray((-0.1, 0.2)),
        0.1,
    )

    assert evaluation.successful
    assert evaluation.evidence.finite
    assert evaluation.candidate_state.tensions == pytest.approx(jnp.zeros((2,)))
    assert evaluation.directional_slip_increment == pytest.approx(jnp.zeros((2, 1)))
    assert evaluation.forward_capstan_margin == pytest.approx(jnp.zeros((1,)))
    assert evaluation.reverse_capstan_margin == pytest.approx(jnp.zeros((1,)))
    assert evaluation.evidence.interface_dissipation == pytest.approx(jnp.zeros((1,)))
    assert jnp.all(jnp.isfinite(evaluation.candidate_state.stress_free_lengths))
    assert jnp.all(jnp.isfinite(evaluation.candidate_state.slip))


def test_prepared_capstan_update_retains_dynamic_bounds_under_filter_jit():
    _, _, route = _rod_and_route()
    state = _state()
    prepared = CapstanTendonFrictionPlan(
        jnp.asarray((log(2.0),)),
        jnp.asarray((1.0,)),
        jnp.asarray((100.0, 100.0)),
    ).prepare(route, state)
    evaluate = eqx.filter_jit(prepared.evaluate)

    evaluation = evaluate(
        state,
        jnp.asarray((1.1, 1.3)),
        jnp.zeros((2,)),
        0.1,
    )

    assert evaluation.prepared_id == prepared.prepared_id
    assert evaluation.successful
    assert evaluation.forward_capstan_margin[0] == pytest.approx(0.0, abs=2.0e-4)


def test_failed_vi_keeps_candidate_evidence_and_rolls_back_all_history():
    _, _, route = _rod_and_route()
    state = _state((7.0, 11.0), slip=0.04)
    termination = NonlinearTermination(
        absolute_residual=0.0,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=1,
        maximum_evaluations=1,
    )
    prepared = CapstanTendonFrictionPlan(
        jnp.asarray((0.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray((100.0, 100.0)),
        termination=termination,
    ).prepare(route, state)

    evaluation = prepared.evaluate(
        state,
        jnp.asarray((1.1, 1.3)),
        jnp.zeros((2,)),
        0.1,
    )

    assert not evaluation.successful
    assert not evaluation.evidence.converged
    assert evaluation.evidence.rollback_applied
    assert evaluation.candidate_state.tensions == pytest.approx(jnp.asarray((10.0, 30.0)))
    assert not jnp.array_equal(
        evaluation.candidate_state.tensions, evaluation.accepted_state.tensions
    )
    assert jnp.array_equal(evaluation.accepted_state.tensions, state.tensions)
    assert jnp.array_equal(
        evaluation.accepted_state.stress_free_lengths, state.stress_free_lengths
    )
    assert jnp.array_equal(evaluation.accepted_state.slip, state.slip)
