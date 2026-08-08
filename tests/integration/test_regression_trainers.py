#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import optax
from evosax import algorithms as evo_algos

import phydrax as phx
from phydrax.nn.models import MLP
from phydrax.solver import FunctionalSolver


def _make_regression_solver(seed: int, *, scan: bool = False) -> FunctionalSolver:
    geom = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geom.Function("x")
    def target(x):
        return x[0] + 2.0 * x[1]

    xs = jnp.linspace(-1.0, 1.0, 6)
    grid = jnp.stack(jnp.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape((-1, 2))

    component = geom.component()
    batch = component.points({"x": grid})
    condition = phx.conditions.Observation("u", component, target)
    source = phx.integration.fixed(
        phx.integration.from_samples(phx.integration.mean_over(component), batch)
    )
    term = phx.terms.ObservationPenalty(condition, source)

    model = MLP(
        in_size=2, out_size="scalar", hidden_sizes=(), scan=scan, key=jr.key(seed)
    )
    u = geom.Model("x")(model)
    return FunctionalSolver(functions={"u": u}, terms=[term])


def test_regression_2d_optax():
    solver = _make_regression_solver(seed=0)
    init_loss = solver.loss(key=jr.key(0))

    trained = solver.solve(
        num_iter=120,
        optim=optax.adam(1e-2),
        seed=0,
        jit=True,
        keep_best=True,
    )
    final_loss = trained.loss(key=jr.key(0))
    assert final_loss < init_loss


def test_regression_2d_optax_scan():
    solver = _make_regression_solver(seed=3, scan=True)
    init_loss = solver.loss(key=jr.key(0))

    trained = solver.solve(
        num_iter=120,
        optim=optax.adam(1e-2),
        seed=0,
        jit=True,
        keep_best=True,
    )
    final_loss = trained.loss(key=jr.key(0))
    assert final_loss < init_loss


def test_regression_2d_evosax():
    solver = _make_regression_solver(seed=1)
    init_loss = solver.loss(key=jr.key(0))

    params = solver.trainable_functions()

    algo_name = "Open_ES"
    algo_dict = vars(evo_algos)
    candidates = [a for a in algo_dict.keys() if not a.startswith("_")]
    s_lower = algo_name.replace("-", "_").lower()
    match = None
    for a in candidates:
        if a.lower() == s_lower:
            match = a
            break
    assert match is not None
    AlgoCls = algo_dict[match]
    algo = AlgoCls(population_size=32, solution=params)

    trained = solver.solve(
        num_iter=40,
        optim=algo,
        seed=0,
        jit=True,
        keep_best=True,
    )
    final_loss = trained.loss(key=jr.key(0))
    assert final_loss < init_loss


def test_regression_2d_optax_lbfgs_linesearch():
    solver = _make_regression_solver(seed=2)
    init_loss = solver.loss(key=jr.key(0))

    trained = solver.solve(
        num_iter=60,
        optim=optax.lbfgs(learning_rate=1.0),
        seed=0,
        jit=True,
        keep_best=True,
    )
    final_loss = trained.loss(key=jr.key(0))
    assert final_loss < init_loss
