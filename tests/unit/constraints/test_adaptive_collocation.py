#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.constraints import (
    CoresetCollocation,
    FunctionalConstraint,
    PeriodicCollocation,
    R3,
    RARD,
)
from phydrax.domain import Interval1d, PointBatch, SampleLayout
from phydrax.sampling import HaltonDesign


def _interval_constraint(
    policy,
    *,
    num_points=32,
    upper=1.0,
    residual_scale=1.0,
    residual_power=1,
):
    domain = Interval1d(0.0, upper)
    structure = SampleLayout((("x",),))

    @domain.Function("x")
    def coordinate(x):
        return residual_scale * (x[0] / upper) ** residual_power

    constraint = FunctionalConstraint.from_operator(
        component=domain.component(),
        operator=lambda _u: coordinate,
        constraint_vars="u",
        sampling=phx.domain.PointSampling(num_points, layout=structure, design="uniform"),
        collocation_policy=policy,
    )
    return domain, constraint, {"u": domain.Function()(0.0)}


def _coordinates(population):
    field = population.batch.points["x"]
    assert isinstance(field, cx.Field)
    return jnp.asarray(field.data).reshape((-1,))


def test_periodic_collocation_replaces_a_fixed_size_population():
    policy = PeriodicCollocation(refresh_every=2, sampler="uniform")
    _domain, constraint, functions = _interval_constraint(policy)
    initial = policy.initialize(constraint, key=jr.key(0))
    refreshed = policy.refresh(constraint, functions, initial, key=jr.key(1), iter_=2)
    assert isinstance(initial.batch, PointBatch)
    assert refreshed.batch.structure == initial.batch.structure
    assert _coordinates(refreshed).shape == _coordinates(initial).shape
    assert not jnp.allclose(_coordinates(refreshed), _coordinates(initial))


def test_collocation_policy_accepts_typed_reference_design():
    policy = R3(
        refresh_every=1,
        sampler=HaltonDesign(scrambled=True),
    )
    _domain, constraint, functions = _interval_constraint(policy)
    population = policy.initialize(constraint, key=jr.key(2))

    refreshed = policy.refresh(
        constraint,
        functions,
        population,
        key=jr.key(3),
        iter_=1,
    )

    assert isinstance(policy.sampler, HaltonDesign)
    assert isinstance(refreshed.batch, PointBatch)


def test_r3_retains_difficult_points_and_preserves_population_size():
    policy = R3(refresh_every=1, sampler="uniform")
    _domain, constraint, functions = _interval_constraint(policy, num_points=64)
    initial = policy.initialize(constraint, key=jr.key(4))
    initial_x = _coordinates(initial)
    refreshed = policy.refresh(constraint, functions, initial, key=jr.key(5), iter_=1)
    refreshed_x = _coordinates(refreshed)
    assert refreshed_x.shape == initial_x.shape
    difficult = initial_x[initial_x * initial_x > jnp.mean(initial_x * initial_x)]
    assert jnp.all(jnp.isin(difficult, refreshed_x))


def test_fixed_capacity_rar_d_activates_new_slots():
    policy = RARD(
        refresh_every=1,
        sampler="uniform",
        initial_active_fraction=0.5,
        refinement_fraction=0.25,
    )
    _domain, constraint, functions = _interval_constraint(policy, num_points=40)
    initial = policy.initialize(constraint, key=jr.key(12))
    assert initial.active is not None
    before = int(jnp.sum(jnp.asarray(initial.active.data)))
    refreshed = policy.refresh(constraint, functions, initial, key=jr.key(13), iter_=1)
    assert refreshed.active is not None
    assert int(jnp.sum(jnp.asarray(refreshed.active.data))) == before + 10
    assert _coordinates(refreshed).shape == (40,)


def test_coreset_collocation_preserves_capacity_and_reports_candidate_cost():
    policy = CoresetCollocation(
        refresh_every=1,
        sampler="halton_scrambled",
        candidate_multiplier=4,
        block_size=8,
    )
    _domain, constraint, functions = _interval_constraint(policy, num_points=16)
    initial = policy.initialize(constraint, key=jr.key(20))

    refreshed = policy.refresh(
        constraint,
        functions,
        initial,
        key=jr.key(21),
        iter_=1,
    )
    metrics = policy.data_metrics(refreshed)

    assert _coordinates(refreshed).shape == (16,)
    assert jnp.unique(_coordinates(refreshed)).shape == (16,)
    assert int(metrics["coreset_candidate_count"]) == 64
    assert int(metrics["coreset_selection_kernel_evaluations"]) == 10_496
    assert int(metrics["coreset_selection_valid"]) == 1
    assert 0.0 < metrics["coreset_importance_effective_sample_size"] <= 64.0
    assert jnp.isfinite(metrics["coreset_selection_mmd"])
    assert int(policy.refresh_residual_evaluations(initial)) == 64
    assert int(refreshed.refresh_count) == 1
    assert int(refreshed.last_refresh) == 1


def test_coreset_defaults_delay_refresh_and_controlled_policy_preserves_activation():
    policy = CoresetCollocation(refresh_every=5)
    _domain, constraint, _functions = _interval_constraint(policy)
    population = policy.initialize(constraint, key=jr.key(30))
    controlled = phx.constraints.controlled_collocation(policy)

    assert policy.start_at == 10
    assert not bool(policy.should_refresh(population, 9))
    assert bool(policy.should_refresh(population, 10))
    assert controlled.schedule.start_at == 10


def test_coreset_importance_is_invariant_to_residual_units():
    policy = CoresetCollocation(
        refresh_every=1,
        start_at=1,
        candidate_multiplier=4,
        uniform_fraction=0.25,
        minimum_ess_fraction=0.25,
        kernel=phx.coresets.RadialKernel(length_scale=0.2),
        max_fill_distance_ratio=10.0,
        block_size=8,
    )
    _domain, base_constraint, base_functions = _interval_constraint(
        policy,
        num_points=16,
    )
    _domain, scaled_constraint, scaled_functions = _interval_constraint(
        policy,
        num_points=16,
        residual_scale=1_000.0,
    )
    base = policy.initialize(base_constraint, key=jr.key(31))
    scaled = policy.initialize(scaled_constraint, key=jr.key(31))

    base_refreshed = policy.refresh(
        base_constraint,
        base_functions,
        base,
        key=jr.key(32),
        iter_=1,
    )
    scaled_refreshed = policy.refresh(
        scaled_constraint,
        scaled_functions,
        scaled,
        key=jr.key(32),
        iter_=1,
    )

    assert jnp.array_equal(_coordinates(base_refreshed), _coordinates(scaled_refreshed))


def test_coreset_auto_scale_is_affine_invariant_and_ess_guard_is_enforced():
    policy = CoresetCollocation(
        refresh_every=1,
        start_at=1,
        candidate_multiplier=4,
        uniform_fraction=0.0,
        minimum_ess_fraction=0.75,
        max_fill_distance_ratio=10.0,
        block_size=8,
    )
    _domain, unit_constraint, unit_functions = _interval_constraint(
        policy,
        num_points=16,
        residual_power=32,
    )
    _domain, scaled_constraint, scaled_functions = _interval_constraint(
        policy,
        num_points=16,
        upper=100.0,
        residual_power=32,
    )
    unit = policy.initialize(unit_constraint, key=jr.key(33))
    scaled = policy.initialize(scaled_constraint, key=jr.key(33))
    unit_refreshed = policy.refresh(
        unit_constraint,
        unit_functions,
        unit,
        key=jr.key(34),
        iter_=1,
    )
    scaled_refreshed = policy.refresh(
        scaled_constraint,
        scaled_functions,
        scaled,
        key=jr.key(34),
        iter_=1,
    )
    unit_metrics = policy.data_metrics(unit_refreshed)
    scaled_metrics = policy.data_metrics(scaled_refreshed)

    assert jnp.allclose(
        _coordinates(unit_refreshed),
        _coordinates(scaled_refreshed) / 100.0,
    )
    assert jnp.allclose(
        unit_metrics["coreset_kernel_length_scale_min"],
        scaled_metrics["coreset_kernel_length_scale_min"],
    )
    assert int(unit_metrics["coreset_kernel_automatic"]) == 1
    assert int(unit_metrics["coreset_ess_guard_triggered"]) == 1
    assert unit_metrics["coreset_effective_uniform_fraction"] > 0.0
    assert unit_metrics["coreset_importance_effective_sample_fraction"] >= 0.75


def test_coreset_fill_distance_guard_retains_the_current_population():
    policy = CoresetCollocation(
        refresh_every=1,
        start_at=1,
        candidate_multiplier=8,
        exponent=1.0,
        uniform_fraction=0.0,
        minimum_ess_fraction=0.01,
        max_fill_distance_ratio=1.0,
        kernel=phx.coresets.RadialKernel(length_scale=0.01),
        block_size=16,
    )
    _domain, constraint, functions = _interval_constraint(
        policy,
        num_points=16,
        residual_power=32,
    )
    initial = policy.initialize(constraint, key=jr.key(40))
    refreshed = policy.refresh(
        constraint,
        functions,
        initial,
        key=jr.key(41),
        iter_=1,
    )
    metrics = policy.data_metrics(refreshed)

    assert jnp.array_equal(_coordinates(refreshed), _coordinates(initial))
    assert int(metrics["coreset_selection_valid"]) == 1
    assert int(metrics["coreset_selection_accepted"]) == 0
    assert int(metrics["coreset_coverage_guard_triggered"]) == 1
    assert (
        metrics["coreset_coverage_fill_distance"]
        > metrics["coreset_coverage_baseline_fill_distance"]
    )
    assert jnp.isfinite(metrics["coreset_selection_mmd"])


def test_coreset_collocation_is_declared_conditional():
    support = phx.constraints.collocation_policy_support(CoresetCollocation())

    assert support.name == "coreset"
    assert support.tier == "conditional"