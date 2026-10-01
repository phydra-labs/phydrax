# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from collections.abc import Iterator
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

import phydrax as phx
from examples.meshfree_learned_edge_flux import (
    _component,
    bind,
    BoundLearnedFluxRecovery,
    ConstitutiveScale,
    prepare_recovery,
    run_workflow,
)
from phydrax._training_kernel import TrainingRejectionBudgetError


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


@pytest.mark.parametrize("dimension,size", [(1, 7), (2, 12), (3, 19)])
def test_real_model_training_recovers_law_and_predicts_unseen_geometry(
    dimension: int, size: int
) -> None:
    metrics = run_workflow(size=size, dimension=dimension, seed=2, steps=40)
    assert metrics["primal_successful"] and metrics["unseen_primal_successful"]
    assert metrics["adjoint_successful"] and metrics["coercivity_certified"]
    assert cast(float, metrics["parameter_error"]) < 2e-4
    assert cast(float, metrics["gradient_relative_error"]) < 2e-5
    assert cast(float, metrics["unseen_cloud_error"]) < 2e-5
    assert cast(float, metrics["conservation_defect"]) < 1e-9
    assert metrics["failure_rejected"] and metrics["coverage_refused"]
    assert cast(int, metrics["accepted_updates"]) > 0


def _unsuccessful_forward(
    owner: BoundLearnedFluxRecovery, case: None
) -> phx.solver.SolverCaseResult:
    del case
    coefficient = -owner.component.model(jnp.zeros(()))
    solved = owner.recovery.prepared.solve(
        parameters={"constitutive-strength": coefficient}
    )
    return phx.solver.SolverCaseResult(
        residual=solved.state - owner.recovery.reference,
        accepted=solved.accepted,
        aux=(solved.primal_status,),
    )


def test_failed_forward_is_rejected_by_objective_and_training_without_parameter_update() -> (
    None
):
    recovery = prepare_recovery(size=9, dimension=2, seed=4)
    initial = _component(ConstitutiveScale(1.0), recovery)
    objective = phx.solver.SolverObjective(
        recovery,
        bind,
        _unsuccessful_forward,
        objective_id="meshfree-reject-failed-forward",
        accepted_results="reject-attempt",
    )
    evaluation = objective.evaluate(initial)
    assert not bool(jnp.all(evaluation.accepted))
    assert not bool(jnp.isfinite(evaluation.value))
    with pytest.raises(TrainingRejectionBudgetError) as raised:
        phx.solver.train_components(
            initial,
            (objective,),
            optimizer=optax.sgd(0.1),
            steps=2,
            rejection_budget=1,
            key=jr.key(4),
        )
    assert int(raised.value.state.accepted_cursor) == 0
    assert int(raised.value.state.nonfinite_rejections) == 2
    assert isinstance(initial.model, ConstitutiveScale)
    np.testing.assert_allclose(initial.model.log_scale, 0.0, atol=0)
