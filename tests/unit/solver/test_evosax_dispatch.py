#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import cast

import jax.numpy as jnp
import jax.random as jr
import pytest
from evosax.algorithms import DifferentialEvolution, Open_ES

import phydrax as phx
from phydrax._training import TrainingIterationKind
from phydrax.solver import FunctionalSolver
from phydrax.solver._functional_backend import FunctionalSolveConfig, solve


_DUMMY_SOLVER = cast(FunctionalSolver, object())


def _scalar_training_solver() -> FunctionalSolver:
    domain = phx.domain.Interval1d(0.0, 1.0)
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(0),
    )
    field = domain.Model("x")(model)
    component = domain.component()
    condition = phx.conditions.Residual(
        "u",
        component,
        lambda current: current - 1.0,
    )
    batch = component.points({"x": jnp.asarray([[0.2], [0.8]])})
    term = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(
            phx.integration.from_samples(
                phx.integration.mean_over(component),
                batch,
            )
        ),
    )
    return FunctionalSolver(functions={"u": field}, terms=(term,))


def test_population_based_evosax_is_rejected_with_search_space_guidance():
    algorithm = DifferentialEvolution(
        population_size=4,
        solution=jnp.zeros((2,)),
    )

    with pytest.raises(
        NotImplementedError,
        match=r"initial population.*DesignConstraintSystem\.search",
    ):
        solve(
            _DUMMY_SOLVER,
            optim=algorithm,
            config=FunctionalSolveConfig(num_iter=1),
        )


def test_unrelated_optimizer_object_is_rejected_before_training():
    with pytest.raises(TypeError, match="Optax transformation"):
        solve(
            _DUMMY_SOLVER,
            optim=object(),
            config=FunctionalSolveConfig(num_iter=1),
        )


def test_evaluation_parameters_remains_optax_only_for_evosax():
    algorithm = Open_ES(
        population_size=8,
        solution=jnp.zeros((2,)),
    )

    with pytest.raises(ValueError, match="only for Optax"):
        solve(
            _DUMMY_SOLVER,
            optim=algorithm,
            config=FunctionalSolveConfig(
                num_iter=1,
                evaluation_parameters=lambda state, parameters: parameters,
            ),
        )


def test_distribution_based_evosax_delivers_cadenced_session_metrics():
    solver = _scalar_training_solver()
    events = []
    session = phx.execution.IterationSession(
        "evosax-session",
        sinks=(
            phx.execution.CallableIterationSink(
                events.append,
                "capture-evosax-events",
            ),
        ),
    )
    algorithm = Open_ES(
        population_size=4,
        solution=solver.trainable_functions(),
    )

    solver.solve(
        num_iter=2,
        optim=algorithm,
        seed=0,
        jit=False,
        keep_best=False,
        log_every=0,
        session=session,
        session_every=2,
    )

    kinds = [TrainingIterationKind(int(event.record.metrics.kind)) for event in events]
    assert kinds == [
        TrainingIterationKind.RUN_START,
        TrainingIterationKind.UPDATE,
        TrainingIterationKind.RUN_TERMINAL,
    ]
    assert int(events[1].record.metrics.update_step) == 2
    assert "train/loss" in events[1].record.metrics.metric_names
