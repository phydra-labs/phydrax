#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import optax
from evosax import algorithms as evo_algos

import phydrax as phx


def test_integral_functional_returns_raw_signed_value():
    geom = phx.domain.Interval1d(0.0, 1.0)

    @geom.Function()
    def density():
        return -2.0

    objective = phx.objectives.IntegralFunctional(
        component=geom.component(),
        integrand=density,
        num_points={"x": phx.domain.LegendreAxisSpec(12)},
        structure=phx.domain.ProductStructure((("x",),)),
        label="negative_energy",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"density": density},
        constraints=(),
        objectives=[objective],
    )

    value = solver.loss(key=jr.key(0))
    assert jnp.allclose(value, -2.0, atol=1e-12)


def test_deep_ritz_objective_optimizes_trainable_field_with_correct_derivative():
    geom = phx.domain.Interval1d(0.0, 1.0)

    @geom.Function("x")
    def coordinate(x):
        return x[0]

    amplitude = geom.Parameter(0.0)
    field = amplitude * coordinate * (1.0 - coordinate)

    def density(functions):
        u = functions["u"]
        gradient = phx.operators.grad(u, var="x")
        gradient_sq = phx.operators.einsum("...i,...i->...", gradient, gradient)
        return 0.5 * gradient_sq - u

    objective = phx.objectives.IntegralFunctional(
        component=geom.component(),
        integrand=density,
        num_points={"x": phx.domain.LegendreAxisSpec(24)},
        structure=phx.domain.ProductStructure((("x",),)),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(1),
        label="deep_ritz_energy",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        constraints=(),
        objectives=[objective],
    )

    trained = solver.solve(
        num_iter=60,
        optim=optax.adam(5e-2),
        seed=2,
        jit=True,
        keep_best=True,
        log_every=0,
    )

    final_energy = trained.loss(key=jr.key(3))
    midpoint = trained["u"].func(jnp.asarray([0.5]))
    quarter_slope = phx.operators.grad(trained["u"], var="x").func(
        jnp.asarray([0.25])
    )
    assert jnp.allclose(final_energy, -1.0 / 24.0, atol=3e-4)
    assert jnp.allclose(midpoint, 0.125, atol=3e-3)
    assert jnp.allclose(quarter_slope, jnp.asarray([0.25]), atol=6e-3)


def test_evosax_optimizes_raw_integral_objective():
    geom = phx.domain.Interval1d(0.0, 1.0)
    field = geom.Parameter(0.0)

    def density(functions):
        return -functions["u"]

    objective = phx.objectives.IntegralFunctional(
        component=geom.component(),
        integrand=density,
        num_points={"x": phx.domain.LegendreAxisSpec(8)},
        structure=phx.domain.ProductStructure((("x",),)),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(4),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        constraints=(),
        objectives=[objective],
    )
    algo = evo_algos.Open_ES(
        population_size=8,
        solution=solver.trainable_functions(),
    )

    trained = solver.solve(
        num_iter=5,
        optim=algo,
        seed=0,
        jit=True,
        keep_best=True,
        log_every=0,
    )
    assert trained.loss(key=jr.key(5)) < 0.0
