#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from phydrax._strict import StrictModule
from phydrax._trainable import parameter_field
from phydrax._training import TrainingIterationKind
from phydrax._training_kernel import (
    KernelObjective,
    prepare_training_kernel,
    run_training_attempt,
    TrainingAttemptOutcome,
    TrainingKernelSpec,
)
from phydrax._training_objective import _ObjectiveContribution
from phydrax.optim._evolution_strategy import (
    DistributionEvolutionPayload,
    DistributionEvolutionUpdateRule,
)
from phydrax.solver import FunctionalSolver
from phydrax.solver._functional_backend import FunctionalSolveConfig, solve


_DUMMY_SOLVER = cast(FunctionalSolver, object())


class _Mean(StrictModule):
    weight: jax.Array = parameter_field()


def _finite_only_at_anchor(parameters, model_state, fixed, payload, keys):
    del fixed, keys
    at_anchor = jnp.all(parameters.weight == payload.objective)
    value = jnp.where(at_anchor, jnp.sum(parameters.weight**2), jnp.nan)
    return _ObjectiveContribution(value, jnp.ones(())), model_state, ()


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


def test_native_evolution_strategy_requires_antithetic_population():
    with pytest.raises(ValueError, match="even integer"):
        phx.optim.OpenEvolutionStrategy(3)


def test_unrelated_optimizer_object_is_rejected_before_training():
    with pytest.raises(TypeError, match="Optax transformation"):
        solve(
            _DUMMY_SOLVER,
            optim=object(),
            config=FunctionalSolveConfig(num_iter=1),
        )


def test_evaluation_parameters_remains_optax_only_for_evolution():
    algorithm = phx.optim.OpenEvolutionStrategy(8)

    with pytest.raises(ValueError, match="only for Optax"):
        solve(
            _DUMMY_SOLVER,
            optim=algorithm,
            config=FunctionalSolveConfig(
                num_iter=1,
                evaluation_parameters=lambda state, parameters: parameters,
            ),
        )


def test_distribution_evolution_delivers_cadenced_session_metrics():
    solver = _scalar_training_solver()
    events = []
    session = phx.execution.IterationSession(
        "evolution-session",
        sinks=(
            phx.execution.CallableIterationSink(
                events.append,
                "capture-evolution-events",
            ),
        ),
    )
    algorithm = phx.optim.OpenEvolutionStrategy(4)

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


def test_distribution_evolution_generation_without_finite_fitness_rolls_back():
    tree = _Mean(jnp.asarray([1.0, -2.0]))
    kernel = prepare_training_kernel(
        tree,
        (
            KernelObjective(
                objective_id="anchored",
                kind=ObjectiveKind.DATA_FIT,
                route=DerivativeRoute.DIRECT,
                fn=_finite_only_at_anchor,
            ),
        ),
        TrainingKernelSpec(
            DistributionEvolutionUpdateRule(
                phx.optim.OpenEvolutionStrategy(4, standard_deviation_decay=0.5),
                jr.key(1),
                rule_id="nonfinite-generation",
            ),
            context="evolution rollback",
            rejection_budget=1,
        ),
        root_authority=ComponentAuthority.MODEL,
    )
    state = kernel.init(tree, jr.key(0))

    # The committed mean is finite; every member of the generation is not.
    rejected, evidence = run_training_attempt(
        kernel, state, DistributionEvolutionPayload(tree.weight, jr.key(2))
    )

    assert int(evidence.outcome) == TrainingAttemptOutcome.NONFINITE
    assert bool(evidence.finite)
    np.testing.assert_array_equal(rejected.parameters.weight, tree.weight)
    algorithm = rejected.rule_state.algorithm
    assert int(algorithm.generation) == 0
    assert float(algorithm.standard_deviation) == 0.1
    assert int(rejected.accepted_cursor) == 0
    assert int(rejected.nonfinite_rejections) == 1
