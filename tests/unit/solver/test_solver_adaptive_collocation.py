#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
from phydrax.constraints import (
    controlled_collocation,
    ControlledCollocationPopulation,
    FunctionalConstraint,
    PeriodicCollocation,
    R3,
    RefreshGuard,
)
from phydrax.domain import Interval1d, SampleLayout
from phydrax.nn import MLP
from phydrax.solver import FunctionalSolver


def _trainable_interval_solver(policy):
    domain = Interval1d(0.0, 1.0)
    structure = SampleLayout((("x",),))
    model = MLP(
        in_size=1,
        out_size="scalar",
        width_size=8,
        depth=1,
        key=jr.key(0),
    )
    u = domain.Model("x")(model)
    constraint = FunctionalConstraint.from_operator(
        component=domain.component(),
        operator=lambda field: field - 1.0,
        constraint_vars="u",
        sampling=phx.domain.PointSampling(16, layout=structure, design="uniform"),
        collocation_policy=policy,
    )
    return FunctionalSolver(functions={"u": u}, constraints=(constraint,))


def test_solver_initializes_and_uses_adaptive_population():
    solver = _trainable_interval_solver(R3(refresh_every=1, sampler="uniform"))
    assert len(solver.collocation) == 1
    assert solver.collocation[0] is not None
    loss = solver.loss(key=jr.key(1), iter_=1)
    assert jnp.isfinite(loss)


def test_solver_returns_updated_collocation_state_after_training():
    solver = _trainable_interval_solver(
        PeriodicCollocation(refresh_every=1, sampler="uniform")
    )
    initial = solver.collocation[0]
    assert initial is not None
    trained = solver.solve(
        num_iter=2,
        optim=optax.adam(1e-3),
        seed=2,
        jit=True,
        keep_best=False,
        log_every=0,
    )
    updated = trained.collocation[0]
    assert updated is not None
    assert int(updated.refresh_count) == 2
    assert int(updated.last_refresh) == 2
    assert not jnp.allclose(
        initial.batch.points["x"].data,
        updated.batch.points["x"].data,
    )


def test_solver_logs_adaptive_population_diagnostics(capsys):
    solver = _trainable_interval_solver(
        PeriodicCollocation(refresh_every=1, sampler="uniform")
    )
    solver.solve(
        num_iter=1,
        optim=optax.adam(1e-3),
        seed=7,
        jit=True,
        keep_best=False,
        log_every=1,
        log_constraints=True,
    )
    output = capsys.readouterr().out
    assert "refresh_count=" in output
    assert "point_count=" in output
    assert "effective_sample_size=" in output


def test_solver_records_controlled_collocation_evaluation_budgets():
    solver = _trainable_interval_solver(
        controlled_collocation(
            PeriodicCollocation(refresh_every=1, sampler="uniform"),
            guard=RefreshGuard(max_relative_regression=1e6),
        )
    )
    trained = solver.solve(
        num_iter=2,
        optim=optax.adam(1e-3),
        seed=8,
        jit=True,
        keep_best=False,
        log_every=0,
    )
    population = trained.collocation[0]

    assert isinstance(population, ControlledCollocationPopulation)
    assert int(population.refresh_attempt_count) == 2
    assert int(population.refresh_accept_count) == 2
    assert int(population.monitor_evaluations) == 48
    assert int(population.training_evaluations) == 32
    assert not bool(population.proposal_pending)


def test_solver_profiles_device_synchronized_adaptive_refresh_boundary():
    solver = _trainable_interval_solver(R3(refresh_every=1, sampler="uniform"))
    trained = solver.solve(
        num_iter=2,
        optim=optax.adam(1e-3),
        seed=9,
        jit=True,
        keep_best=False,
        log_every=0,
        profile_adaptive=True,
    )

    diagnostics = trained.training_diagnostics
    assert bool(diagnostics["profile_enabled"])
    assert float(diagnostics["refresh_wall_time_seconds"]) > 0.0
    assert float(diagnostics["optimizer_wall_time_seconds"]) > 0.0
