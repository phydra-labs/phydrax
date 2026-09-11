from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import phydrax as phx
from phydrax.optim import (
    conflict_free_gradient,
    ConflictFreeGradientPolicy,
    ConflictFreeGradientStatus,
    ConflictFreeUpdatePolicy,
)


def test_orthogonal_gradients_receive_positive_equal_projections():
    result = conflict_free_gradient((jnp.asarray((1.0, 0.0)), jnp.asarray((0.0, 1.0))))

    np.testing.assert_allclose(result.direction, (1.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(result.projections, (1.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(result.cosine_matrix, jnp.eye(2), atol=1e-6)
    assert bool(result.successful)
    assert int(result.status) == int(ConflictFreeGradientStatus.SUCCESS)


def test_conflicting_feasible_gradients_still_share_a_descent_direction():
    gradients = (
        {"weight": jnp.asarray((1.0, 0.0))},
        {"weight": jnp.asarray((-0.5, 1.0))},
    )
    result = conflict_free_gradient(gradients)

    assert bool(result.successful)
    assert bool(jnp.all(result.projections > 0.0))
    assert not bool(jnp.any(result.conflicts))


def test_opposite_gradients_fail_without_a_false_conflict_free_claim():
    result = conflict_free_gradient((jnp.asarray((1.0, 0.0)), jnp.asarray((-1.0, 0.0))))

    assert not bool(result.successful)
    assert int(result.status) == int(ConflictFreeGradientStatus.INFEASIBLE)
    np.testing.assert_allclose(result.direction, 0.0, atol=1e-7)
    with pytest.raises(eqx.EquinoxRuntimeError, match="conflict-free"):
        checked = conflict_free_gradient(
            (jnp.asarray((1.0, 0.0)), jnp.asarray((-1.0, 0.0))),
            policy=ConflictFreeGradientPolicy(failure="error"),
        )
        np.asarray(checked.direction)


def test_rank_deficient_identical_gradients_remain_usable():
    result = conflict_free_gradient((jnp.asarray((2.0, -1.0)), jnp.asarray((2.0, -1.0))))

    assert int(result.rank) == 1
    assert bool(result.successful)
    assert bool(jnp.all(result.projections > 0.0))


def test_stationary_inactive_and_complex_objectives_are_distinct():
    result = conflict_free_gradient(
        (
            {"z": jnp.asarray((1.0 + 1.0j,))},
            {"z": jnp.asarray((1.0 - 1.0j,))},
            {"z": jnp.asarray((0.0 + 0.0j,))},
        ),
        active=jnp.asarray((True, True, False)),
    )

    assert bool(result.successful)
    assert not bool(result.active[2])
    assert not bool(result.stationary[2])
    np.testing.assert_allclose(result.projections[:2], (2.0, 2.0), atol=1e-6)

    stationary = conflict_free_gradient((jnp.zeros((2,)),))
    assert bool(stationary.successful)
    assert bool(stationary.stationary[0])
    assert int(stationary.status) == int(ConflictFreeGradientStatus.STATIONARY)


def test_structure_shape_nonfinite_and_active_contracts_are_checked():
    with pytest.raises(ValueError, match="structures"):
        conflict_free_gradient(({"a": jnp.ones(1)}, {"b": jnp.ones(1)}))
    with pytest.raises(ValueError, match="shapes"):
        conflict_free_gradient((jnp.ones(1), jnp.ones(2)))
    with pytest.raises(ValueError, match="one Boolean"):
        conflict_free_gradient((jnp.ones(1),), active=jnp.ones((2,), dtype=bool))

    nonfinite = conflict_free_gradient((jnp.asarray((jnp.nan,)),))
    assert not bool(nonfinite.successful)
    assert int(nonfinite.status) == int(ConflictFreeGradientStatus.NONFINITE)


class _ScaledOperator(phx.nn.operator.AbstractOperatorModel):
    gain: jax.Array
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, gain: float):
        self.gain = jnp.asarray(gain)
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        values = batch.input("state").values
        assert values is not None
        return self.gain * values

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _fixed_additive_update(value: float):
    def initialize(parameters):
        del parameters
        return ()

    def update(gradients, state, parameters=None):
        del parameters
        updates = jax.tree.map(
            lambda leaf: jnp.full_like(leaf, value),
            gradients,
        )
        return updates, state

    return optax.GradientTransformation(initialize, update)


def _two_term_functional_solver():
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(jnp.asarray([1.0]))
    component = domain.component()
    batch = component.points({"x": jnp.asarray([[0.25], [0.75]])})
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component),
        batch,
    )
    condition = phx.conditions.Residual("u", component, lambda value: value)
    return phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(
            phx.terms.ResidualPenalty(
                condition,
                phx.integration.fixed(realization),
                label="first",
            ),
            phx.terms.ResidualPenalty(
                condition,
                phx.integration.fixed(realization),
                label="second",
            ),
        ),
    )


def test_functional_solver_composes_one_prepared_objective_vector():
    optimizer = optax.sgd(0.1)
    baseline = _two_term_functional_solver().solve(
        num_iter=1,
        optim=optimizer,
        keep_best=False,
        log_every=0,
        jit=False,
        training=phx.solver.FunctionalTrainingPlan(),
    )
    composed = _two_term_functional_solver().solve(
        num_iter=1,
        optim=optimizer,
        keep_best=False,
        log_every=0,
        jit=False,
        training=phx.solver.FunctionalTrainingPlan(
            gradient_composition=ConflictFreeGradientPolicy()
        ),
    )

    np.testing.assert_allclose(
        composed.functions["u"].func(),
        baseline.functions["u"].func(),
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="gradient accumulation"):
        _two_term_functional_solver().solve(
            num_iter=1,
            optim=optimizer,
            keep_best=False,
            log_every=0,
            gradient_accumulation=2,
            training=phx.solver.FunctionalTrainingPlan(
                gradient_composition=ConflictFreeGradientPolicy()
            ),
        )


def test_functional_solver_aligns_the_applied_optimizer_proposal():
    trained = _two_term_functional_solver().solve(
        num_iter=1,
        optim=_fixed_additive_update(0.1),
        keep_best=False,
        log_every=0,
        jit=True,
        training=phx.solver.FunctionalTrainingPlan(
            gradient_composition=ConflictFreeGradientPolicy(),
            update_alignment=ConflictFreeUpdatePolicy(),
        ),
    )

    np.testing.assert_allclose(trained.functions["u"].func(), (1.0,), atol=1e-10)
    assert trained.training_state is not None
    statistics = trained.training_state.update_alignment_statistics
    assert statistics is not None
    assert float(statistics.proposal_conflict_rate) == 1.0
    assert float(statistics.applied_conflict_rate) == 0.0
    assert float(statistics.projection_rate) == 1.0
    assert (
        float(
            trained.training_diagnostics[
                "optimizer/update_alignment/proposal_conflict_rate"
            ]
        )
        == 1.0
    )
    with pytest.raises(ValueError, match="gradient accumulation"):
        _two_term_functional_solver().solve(
            num_iter=1,
            optim=optax.sgd(0.1),
            keep_best=False,
            log_every=0,
            gradient_accumulation=2,
            training=phx.solver.FunctionalTrainingPlan(
                update_alignment=ConflictFreeUpdatePolicy()
            ),
        )


def test_operator_fit_composes_explicit_loss_terms():
    axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 4))
    values = jnp.stack((axis.nodes, axis.nodes + 1.0), axis=0)
    dataset = phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"solution": 2.0 * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )
    terms = (
        phx.nn.operator.training.SupervisedOperatorLoss(name="first"),
        phx.nn.operator.training.SupervisedOperatorLoss(name="second"),
    )
    result = phx.nn.operator.training.fit_operator(
        _ScaledOperator(0.0),
        dataset,
        loss_terms=terms,
        include_model_losses=False,
        gradient_composition=ConflictFreeGradientPolicy(),
        optimizer=optax.sgd(0.05),
        optimizer_id="sgd",
        epochs=1,
        batch_size=2,
        shuffle=False,
        output_field_map={"output": "solution"},
        jit=False,
    )

    assert float(result.execution_model.gain) > 0.0
    assert result.final_loss < result.initial_loss
    with pytest.raises(ValueError, match="attached model losses"):
        phx.nn.operator.training.fit_operator(
            _ScaledOperator(0.0),
            dataset,
            loss_terms=terms,
            gradient_composition=ConflictFreeGradientPolicy(),
            optimizer=optax.sgd(0.05),
            optimizer_id="sgd",
            epochs=1,
            batch_size=2,
            shuffle=False,
            output_field_map={"output": "solution"},
            jit=False,
        )


def test_operator_fit_aligns_an_ordinary_aggregate_optimizer_proposal():
    axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 4))
    values = jnp.stack((axis.nodes, axis.nodes + 1.0), axis=0)
    dataset = phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"solution": 2.0 * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )
    terms = (
        phx.nn.operator.training.SupervisedOperatorLoss(name="first"),
        phx.nn.operator.training.SupervisedOperatorLoss(name="second"),
    )

    result = phx.nn.operator.training.fit_operator(
        _ScaledOperator(0.0),
        dataset,
        loss_terms=terms,
        include_model_losses=False,
        update_alignment=ConflictFreeUpdatePolicy(),
        optimizer=_fixed_additive_update(-0.1),
        optimizer_id="fixed-harmful-update",
        epochs=1,
        steps=1,
        batch_size=2,
        shuffle=False,
        output_field_map={"output": "solution"},
        jit=True,
    )

    np.testing.assert_allclose(result.last_execution_model.gain, 0.0, atol=1e-10)
    statistics = result.update_alignment_statistics
    assert statistics is not None
    assert float(statistics.proposal_conflict_rate) == 1.0
    assert float(statistics.applied_conflict_rate) == 0.0
    assert result.history.train_metrics[0]["update_alignment/raw_conflict"] == 1.0
    assert result.history.train_metrics[0]["update_alignment/applied_conflict"] == 0.0
    with pytest.raises(ValueError, match="attached model losses"):
        phx.nn.operator.training.fit_operator(
            _ScaledOperator(0.0),
            dataset,
            loss_terms=terms,
            update_alignment=ConflictFreeUpdatePolicy(),
            epochs=1,
            batch_size=2,
        )
    with pytest.raises(ValueError, match="gradient accumulation"):
        phx.nn.operator.training.fit_operator(
            _ScaledOperator(0.0),
            dataset,
            loss_terms=terms,
            include_model_losses=False,
            update_alignment=ConflictFreeUpdatePolicy(),
            gradient_accumulation=2,
            epochs=1,
            batch_size=1,
        )
