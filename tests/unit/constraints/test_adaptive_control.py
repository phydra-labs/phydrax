#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.domain import Interval1d, SampleLayout
from phydrax.sampling.collocation import (
    AdaptationBudget,
    collocation_policy_support,
    controlled_collocation,
    ControlledCollocationPopulation,
    CoresetCollocation,
    CoverageAnchors,
    PeriodicCollocation,
    R3,
    RARD,
    RefreshGuard,
    RefreshSchedule,
)


def _controlled_interval(*, anchors=0.0, budget=None, guard=None):
    domain = Interval1d(0.0, 1.0)
    structure = SampleLayout((("x",),))
    policy = controlled_collocation(
        R3(
            refresh_every=1,
            sampler="uniform",
            min_replace_fraction=0.25,
            max_retain_fraction=0.75,
        ),
        schedule=RefreshSchedule(1),
        guard=RefreshGuard() if guard is None else guard,
        budget=AdaptationBudget() if budget is None else budget,
        anchors=CoverageAnchors(anchors),
    )
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda field: field)
    source = phx.integration.adaptive(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(16, layout=structure, design="uniform"),
        policy,
    )
    term = phx.terms.ResidualPenalty(condition, source)
    return domain, term, policy


def _x(population):
    return population.current.batch.points["x"].data


def test_policy_support_tiers_cover_retained_methods():
    assert collocation_policy_support(None).tier == "stable"
    assert collocation_policy_support(PeriodicCollocation()).tier == "stable"
    assert collocation_policy_support(R3()).tier == "stable"
    assert collocation_policy_support(RARD()).tier == "conditional"
    assert collocation_policy_support("hierarchical_axes").tier == "conditional"


def test_controlled_population_uses_a_fixed_independent_monitor():
    domain, constraint, policy = _controlled_interval()
    population = policy.initialize(constraint, key=jr.key(1))
    assert isinstance(population, ControlledCollocationPopulation)
    monitor_before = population.monitor_batch.points["x"].data
    functions = {"u": domain.Function()(0.0)}

    refreshed = policy.refresh(
        constraint,
        functions,
        population,
        key=jr.key(2),
        iter_=1,
    )

    assert jnp.array_equal(
        jnp.asarray(refreshed.monitor_batch.points["x"].data), monitor_before
    )
    assert int(refreshed.refresh_attempt_count) == 1
    assert int(refreshed.monitor_evaluations) == 16
    assert int(refreshed.candidate_evaluations) == 16
    assert bool(refreshed.proposal_pending)


def test_validation_guard_rolls_back_a_regressing_population():
    domain, constraint, policy = _controlled_interval(
        guard=RefreshGuard(
            max_relative_regression=0.0,
            max_consecutive_rejections=1,
            suspension_steps=7,
        )
    )
    initial = policy.initialize(constraint, key=jr.key(3))
    baseline_x = _x(initial)
    proposed = policy.refresh(
        constraint,
        {"u": domain.Function()(0.0)},
        initial,
        key=jr.key(4),
        iter_=1,
    )
    rejected = policy.refresh(
        constraint,
        {"u": domain.Function()(1.0)},
        proposed,
        key=jr.key(5),
        iter_=2,
    )

    assert jnp.array_equal(_x(rejected), baseline_x)
    assert int(rejected.refresh_reject_count) == 1
    assert int(rejected.refresh_attempt_count) == 1
    assert int(rejected.suspended_until) == 9
    assert not bool(policy.should_refresh(rejected, 8))
    assert bool(policy.should_refresh(rejected, 9))


def test_terminal_settlement_rolls_back_without_admitting_another_proposal():
    domain, constraint, policy = _controlled_interval()
    initial = policy.initialize(constraint, key=jr.key(15))
    baseline_x = _x(initial)
    proposed = policy.refresh(
        constraint,
        {"u": domain.Function()(0.0)},
        initial,
        key=jr.key(16),
        iter_=1,
    )
    settled = policy.settle(
        constraint,
        {"u": domain.Function()(1.0)},
        proposed,
        key=jr.key(17),
        iter_=2,
    )

    assert jnp.array_equal(_x(settled), baseline_x)
    assert int(settled.refresh_attempt_count) == 1
    assert int(settled.refresh_reject_count) == 1
    assert not bool(settled.proposal_pending)


def test_coverage_anchors_survive_population_proposals():
    domain, constraint, policy = _controlled_interval(anchors=0.25)
    initial = policy.initialize(constraint, key=jr.key(6))
    anchors = _x(initial)[:4]
    refreshed = policy.refresh(
        constraint,
        {"u": domain.Function("x")(lambda x: x[0])},
        initial,
        key=jr.key(7),
        iter_=1,
    )

    assert jnp.array_equal(_x(refreshed)[:4], anchors)
    assert float(policy.data_metrics(refreshed)["control_anchor_fraction"]) == 0.25


def test_budget_prevents_refresh_before_overspending_candidate_scores():
    _domain, constraint, policy = _controlled_interval(
        budget=AdaptationBudget(max_candidate_evaluations=15)
    )
    population = policy.initialize(constraint, key=jr.key(8))

    assert not bool(policy.should_refresh(population, 1))


def test_exhausted_candidate_budget_still_validates_the_last_proposal():
    domain, constraint, policy = _controlled_interval(
        budget=AdaptationBudget(max_candidate_evaluations=16)
    )
    initial = policy.initialize(constraint, key=jr.key(9))
    functions = {"u": domain.Function()(0.0)}
    proposed = policy.refresh(
        constraint,
        functions,
        initial,
        key=jr.key(10),
        iter_=1,
    )

    assert bool(proposed.proposal_pending)
    assert int(proposed.candidate_evaluations) == 16
    assert bool(policy.should_refresh(proposed, 2))

    settled = policy.refresh(
        constraint,
        functions,
        proposed,
        key=jr.key(11),
        iter_=2,
    )
    assert not bool(settled.proposal_pending)
    assert int(settled.refresh_accept_count) == 1
    assert int(settled.refresh_attempt_count) == 1
    assert int(settled.candidate_evaluations) == 16
    assert int(settled.monitor_evaluations) == 32


def test_monitor_budget_reserves_validation_for_every_proposal():
    domain, constraint, policy = _controlled_interval(
        budget=AdaptationBudget(max_monitor_evaluations=32)
    )
    initial = policy.initialize(constraint, key=jr.key(12))
    functions = {"u": domain.Function()(0.0)}

    assert bool(policy.should_refresh(initial, 1))
    proposed = policy.refresh(
        constraint,
        functions,
        initial,
        key=jr.key(13),
        iter_=1,
    )
    assert bool(policy.should_refresh(proposed, 2))

    settled = policy.refresh(
        constraint,
        functions,
        proposed,
        key=jr.key(14),
        iter_=2,
    )
    assert not bool(settled.proposal_pending)
    assert int(settled.monitor_evaluations) == 32
    assert not bool(policy.should_refresh(settled, 3))


def test_controlled_coreset_policy_preserves_selection_metrics_through_anchors():
    domain = Interval1d(0.0, 1.0)
    structure = SampleLayout((("x",),))
    policy = controlled_collocation(
        CoresetCollocation(
            refresh_every=1,
            sampler="halton_scrambled",
            candidate_multiplier=3,
            block_size=8,
        ),
        schedule=RefreshSchedule(1),
        anchors=CoverageAnchors(0.25),
    )
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda field: field)
    source = phx.integration.adaptive(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(16, layout=structure, design="uniform"),
        policy,
    )
    term = phx.terms.ResidualPenalty(condition, source)
    initial = policy.initialize(term, key=jr.key(30))
    anchors = _x(initial)[:4]

    refreshed = policy.refresh(
        term,
        {"u": domain.Function("x")(lambda x: x[0])},
        initial,
        key=jr.key(31),
        iter_=1,
    )
    metrics = policy.data_metrics(refreshed)

    assert jnp.array_equal(_x(refreshed)[:4], anchors)
    assert int(metrics["coreset_candidate_count"]) == 48
    assert jnp.isfinite(metrics["coreset_selection_mmd"])
