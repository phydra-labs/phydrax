#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
from phydrax.discretization import NestedDyadicAxisSpec, TensorGridPlan
from phydrax.domain import GridBatch, HyperRectangle
from phydrax.sampling.collocation import (
    HierarchicalAxisCollocation,
    PeriodicSeparableCollocation,
    SeparableCollocationPopulation,
)
from phydrax.solver import FunctionalSolver


def _square_constraint(policy, *, counts=(12, 10)):
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @domain.Function("x")
    def shifted_x(x):
        x0, _x1 = x
        return x0 + 1.0

    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda _u: shifted_x)
    source = phx.integration.adaptive(
        phx.integration.mean_over(component),
        phx.domain.GridSampling({"x": counts}, design="uniform"),
        policy,
    )
    term = phx.terms.ResidualPenalty(condition, source)
    functions = {"u": domain.Function()(0.0)}
    return domain, term, functions


def _axis(population, index):
    values = population.batch.points["x"]
    assert isinstance(values, tuple)
    return jnp.asarray(values[index].data)


def test_separable_population_tracks_logical_and_active_counts():
    policy = PeriodicSeparableCollocation(refresh_every=2)
    domain = phx.domain.GeometryDomain(phx.geometry.Circle((0.0, 0.0), 1.0).compile())
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda _u: domain.Function()(1.0))
    source = phx.integration.adaptive(
        phx.integration.mean_over(component),
        phx.domain.GridSampling({"x": (16, 14)}),
        policy,
    )
    term = phx.terms.ResidualPenalty(condition, source)
    population = policy.initialize(term, key=jr.key(0))
    assert isinstance(population, SeparableCollocationPopulation)
    assert int(population.logical_point_count) == 16 * 14
    assert 0 < int(population.active_logical_point_count) < 16 * 14
    assert tuple(population.axis_age_by_axis) == population.batch.coord_axes_by_label["x"]


def test_periodic_separable_refresh_preserves_shape_and_changes_axes():
    policy = PeriodicSeparableCollocation(refresh_every=1)
    _domain, constraint, functions = _square_constraint(policy)
    initial = policy.initialize(constraint, key=jr.key(1))
    refreshed = policy.refresh(
        constraint,
        functions,
        initial,
        key=jr.key(2),
        iter_=1,
    )
    assert _axis(refreshed, 0).shape == _axis(initial, 0).shape
    assert _axis(refreshed, 1).shape == _axis(initial, 1).shape
    assert not jnp.allclose(_axis(refreshed, 0), _axis(initial, 0))
    assert int(refreshed.refresh_count) == 1
    assert int(refreshed.last_refresh) == 1


def test_nested_dyadic_axis_materializes_fixed_capacity_metadata():
    discretization = NestedDyadicAxisSpec(9, initial_level=1).materialize(
        jnp.asarray(-1.0),
        jnp.asarray(1.0),
    )
    assert discretization.active is not None
    assert discretization.level is not None
    assert discretization.parent_interval is not None
    assert discretization.quad_weights is not None
    assert int(jnp.sum(discretization.active)) == 3
    assert jnp.allclose(jnp.sum(discretization.quad_weights), 2.0)
    activated = discretization.with_active(
        jnp.asarray(discretization.active).at[2].set(True)
    )
    assert activated.active is not None
    assert activated.quad_weights is not None
    assert int(jnp.sum(activated.active)) == 4
    assert jnp.allclose(jnp.sum(activated.quad_weights), 2.0)


def test_hierarchical_axes_activate_nested_nodes_without_shape_changes():
    policy = HierarchicalAxisCollocation(
        refresh_every=1,
        refinement_fraction=0.25,
    )
    domain = HyperRectangle(
        jnp.asarray((-1.0, -1.0)),
        jnp.asarray((1.0, 1.0)),
    )

    @domain.Function("x")
    def shifted_x(x):
        x0, _x1 = x
        return x0 + 1.0

    spec = NestedDyadicAxisSpec(9, initial_level=1)
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda _u: shifted_x)
    source = phx.integration.adaptive(
        phx.integration.over(component),
        phx.domain.GridSampling({"x": TensorGridPlan((spec, spec))}),
        policy,
    )
    term = phx.terms.ResidualPenalty(condition, source)
    functions = {"u": domain.Function()(0.0)}
    initial = policy.initialize(term, key=jr.key(20))
    assert int(initial.logical_point_count) == 81
    assert int(initial.active_logical_point_count) == 1
    refreshed = policy.refresh(
        term,
        functions,
        initial,
        key=jr.key(21),
        iter_=1,
    )
    assert _axis(refreshed, 0).shape == (9,)
    assert _axis(refreshed, 1).shape == (9,)
    assert int(refreshed.active_logical_point_count) == 9
    for discretization in refreshed.batch.axis_discretization_by_axis.values():
        assert discretization.quad_weights is not None
        assert jnp.allclose(jnp.sum(discretization.quad_weights), 2.0)


def test_solver_trains_with_separable_population():
    policy = PeriodicSeparableCollocation(refresh_every=1)
    domain, term, functions = _square_constraint(policy, counts=(6, 5))
    solver = FunctionalSolver(functions=functions, terms=[term])
    trained = solver.solve(
        num_iter=2,
        optim=optax.adam(1e-3),
        seed=9,
        jit=True,
        log_every=0,
    )
    population = trained.collocation[0]
    assert isinstance(population, SeparableCollocationPopulation)
    assert isinstance(population.batch, GridBatch)
    assert int(population.refresh_count) == 2
    assert trained.functions["u"].domain == domain
