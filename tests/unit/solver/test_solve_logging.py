#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from phydrax.constraints import (
    DiscreteInteriorDataConstraint,
    DiscreteTimeDataConstraint,
    SupervisedDatasetConstraint,
)
from phydrax.domain import DatasetDomain, HyperRectangle, TimeInterval
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


def _make_dataset_solver_with_eval(seed: int = 0) -> FunctionalSolver:
    rows = jnp.linspace(0.0, 1.0, 6).reshape((-1, 1))
    domain = DatasetDomain(rows)
    targets = 1.0 + 2.0 * rows[:, 0]
    model = MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(seed),
    )
    u = domain.Model("data")(model)
    train = SupervisedDatasetConstraint(
        "u",
        domain.component(),
        targets,
        num_cases=8,
        indices=jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        label="train_data",
    )
    eval_data = SupervisedDatasetConstraint(
        "u",
        domain.component(),
        targets,
        num_cases=8,
        indices=jnp.asarray([4, 5], dtype=jnp.int32),
        label="eval_data",
    )
    return FunctionalSolver(
        functions={"u": u},
        constraints=[train],
        eval_constraints=[eval_data],
    )


def _make_dataset_solver_with_two_train_constraints(seed: int = 0) -> FunctionalSolver:
    rows = jnp.linspace(0.0, 1.0, 8).reshape((-1, 1))
    domain = DatasetDomain(rows)
    targets = 1.0 + 2.0 * rows[:, 0]
    model = MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(seed),
    )
    u = domain.Model("data")(model)
    train_a = SupervisedDatasetConstraint(
        "u",
        domain.component(),
        targets,
        num_cases=4,
        indices=jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        label="train_a",
    )
    train_b = SupervisedDatasetConstraint(
        "u",
        domain.component(),
        targets,
        num_cases=4,
        indices=jnp.asarray([4, 5, 6, 7], dtype=jnp.int32),
        label="train_b",
    )
    return FunctionalSolver(functions={"u": u}, constraints=[train_a, train_b])


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


def test_solve_text_log_includes_eval_constraints(tmp_path):
    solver = _make_dataset_solver_with_eval()
    log_path = tmp_path / "train_eval.log"

    solver.solve(
        num_iter=2,
        optim=optax.adam(1e-2),
        seed=0,
        log_every=1,
        log_path=log_path,
    )

    text = log_path.read_text(encoding="utf-8")
    assert "[train 0] train_data:" in text
    assert "[eval 0] eval_data:" in text
    assert "data_accuracy=" in text


def test_solve_can_subsample_train_constraints_and_log_all_constraints(tmp_path):
    solver = _make_dataset_solver_with_two_train_constraints()
    log_path = tmp_path / "train_subset.log"

    solver.solve(
        num_iter=2,
        optim=optax.adam(1e-2),
        seed=0,
        log_every=1,
        log_path=log_path,
        train_constraint_sample_size=1,
    )

    text = log_path.read_text(encoding="utf-8")
    assert "[train 0] train_a:" in text
    assert "[train 1] train_b:" in text


def test_solve_can_subsample_train_constraints_without_constraint_logging():
    solver = _make_dataset_solver_with_two_train_constraints()

    solver.solve(
        num_iter=2,
        optim=optax.adam(1e-2),
        seed=0,
        log_every=0,
        log_constraints=False,
        train_constraint_sample_size=1,
    )


def test_solve_rejects_invalid_train_constraint_sample_size():
    solver = _make_dataset_solver_with_two_train_constraints()

    with pytest.raises(ValueError, match="train_constraint_sample_size"):
        solver.solve(
            num_iter=1,
            optim=optax.adam(1e-2),
            train_constraint_sample_size=0,
        )


def test_solver_loss_excludes_eval_constraints():
    domain = DatasetDomain(jnp.asarray([[0.0], [1.0], [2.0]]))
    train_targets = jnp.asarray([0.0, 1.0, 2.0])
    eval_targets = jnp.asarray([10.0, 10.0, 10.0])

    @domain.Function("data")
    def u(data):
        return data[0]

    train = SupervisedDatasetConstraint(
        "u",
        domain.component(),
        train_targets,
        num_cases=8,
    )
    eval_data = SupervisedDatasetConstraint(
        "u",
        domain.component(),
        eval_targets,
        num_cases=8,
    )
    solver = FunctionalSolver(
        functions={"u": u},
        constraints=[train],
        eval_constraints=[eval_data],
    )

    assert jnp.allclose(solver.loss(key=jr.key(4)), 0.0, atol=1e-12)


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
    assert "train/constraints/000_data/loss" in scalar_tags
    assert "train/constraints/000_data/data_accuracy" in scalar_tags
    assert "constraints/000_data/loss" in scalar_tags
    assert "constraints/000_data/data_accuracy" in scalar_tags
    assert "constraints/000_data/data_relative_l2_error" in scalar_tags
    assert "constraints/000_data/data_rmse" in scalar_tags


def test_solve_tensorboard_log_includes_eval_metrics(tmp_path):
    solver = _make_dataset_solver_with_eval()
    log_dir = tmp_path / "tb_eval"

    solver.solve(
        num_iter=2,
        optim=optax.adam(1e-2),
        seed=0,
        log_every=0,
        tensorboard_log_dir=log_dir,
        tensorboard_every=1,
    )

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])

    assert "train/constraints/000_train_data/loss" in scalar_tags
    assert "eval/constraints/000_eval_data/loss" in scalar_tags
    assert "eval/constraints/000_eval_data/data_accuracy" in scalar_tags
    assert "eval/constraints/000_eval_data/data_relative_l2_error" in scalar_tags
    assert "eval/constraints/000_eval_data/data_rmse" in scalar_tags
