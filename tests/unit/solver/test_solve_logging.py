#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import optax
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from phydrax.constraints import DiscreteInteriorDataConstraint, DiscreteTimeDataConstraint
from phydrax.domain import HyperRectangle, TimeInterval
from phydrax.nn import MLP
from phydrax.solver import FunctionalSolver


def _make_supervised_solver(seed: int = 0) -> FunctionalSolver:
    domain = HyperRectangle(jnp.asarray([0.0]), jnp.asarray([1.0]), label="x")
    points = jnp.linspace(0.0, 1.0, 5).reshape((-1, 1))
    values = 1.0 + 2.0 * points[:, 0]

    model = MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(seed),
    )
    u = domain.Model("x")(model)
    data = DiscreteInteriorDataConstraint(
        "u",
        domain,
        points={"x": points},
        values=values,
        label="data",
    )
    return FunctionalSolver(functions={"u": u}, constraints=[data])


def test_discrete_interior_data_constraint_reports_exact_data_metrics():
    domain = HyperRectangle(jnp.asarray([0.0]), jnp.asarray([1.0]), label="x")
    points = jnp.linspace(0.0, 1.0, 5).reshape((-1, 1))
    values = 1.0 + 2.0 * points[:, 0]

    @domain.Function("x")
    def exact(x):
        return 1.0 + 2.0 * x[0]

    constraint = DiscreteInteriorDataConstraint(
        "u",
        domain,
        points={"x": points},
        values=values,
        label="data",
    )

    metrics = constraint.data_metrics({"u": exact}, key=jr.key(0))
    assert jnp.allclose(metrics["data_accuracy"], 1.0)
    assert jnp.allclose(metrics["data_relative_l2_error"], 0.0)
    assert jnp.allclose(metrics["data_rmse"], 0.0)


def test_discrete_time_data_constraint_reports_exact_data_metrics():
    time = TimeInterval(0.0, 1.0)

    @time.Function("t")
    def exact(t):
        return t**2

    times = jnp.linspace(0.0, 1.0, 5)
    values = times**2
    constraint = DiscreteTimeDataConstraint("u", time, times=times, values=values)

    metrics = constraint.data_metrics({"u": exact}, key=jr.key(0))
    assert jnp.allclose(metrics["data_accuracy"], 1.0)
    assert jnp.allclose(metrics["data_relative_l2_error"], 0.0)
    assert jnp.allclose(metrics["data_rmse"], 0.0)


def test_solve_text_log_includes_discrete_data_metrics(tmp_path):
    solver = _make_supervised_solver()
    log_path = tmp_path / "train.log"

    solver.solve(
        num_iter=2,
        optim=optax.adam(1e-2),
        seed=0,
        log_every=1,
        log_path=log_path,
    )

    text = log_path.read_text(encoding="utf-8")
    assert "data_accuracy=" in text
    assert "data_relative_l2_error=" in text
    assert "data_rmse=" in text


def test_solve_tensorboard_log_includes_loss_and_data_metrics(tmp_path):
    solver = _make_supervised_solver()
    log_dir = tmp_path / "tb"

    solver.solve(
        num_iter=2,
        optim=optax.adam(1e-2),
        seed=0,
        log_every=0,
        tensorboard_log_dir=log_dir,
        tensorboard_every=1,
    )

    event_files = tuple(log_dir.glob("events.out.tfevents.*"))
    assert event_files

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])

    assert "train/loss" in scalar_tags
    assert "train/best_loss" in scalar_tags
    assert "train/iter_time_s" in scalar_tags
    assert "constraints/000_data/loss" in scalar_tags
    assert "constraints/000_data/data_accuracy" in scalar_tags
    assert "constraints/000_data/data_relative_l2_error" in scalar_tags
    assert "constraints/000_data/data_rmse" in scalar_tags
