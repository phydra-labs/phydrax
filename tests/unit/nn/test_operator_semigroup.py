#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.operator.training import (
    conditioned_semigroup_consistency_loss,
    ConditionedSemigroupObjective,
)


class _ConditionedTransition(eqx.Module):
    rate: jax.Array
    exact: bool = eqx.field(static=True)

    def __call__(self, batch, *, key=None):
        del key
        state = jnp.asarray(batch.input("state").values)
        duration = jnp.asarray(batch.input("duration").values)
        if state.ndim == duration.ndim + 1:
            duration = duration[..., None]
        if self.exact:
            factor = jnp.exp(self.rate * duration)
        else:
            factor = 1.0 + self.rate * duration
        return factor * state


class _KeyedTransition(eqx.Module):
    def __call__(self, batch, *, key=None):
        state = jnp.asarray(batch.input("state").values)
        duration = jnp.asarray(batch.input("duration").values)
        draw = jr.uniform(key, ())
        return state + duration * draw


def _batch(*, channels=None):
    cases = 3
    points = 4
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, points),
        quadrature_weights=jnp.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    base = 1.0 + jnp.arange(cases * points, dtype=float).reshape(cases, points) / 5.0
    state = base if channels is None else jnp.stack((base, 2.0 * base), axis=-1)
    duration = jnp.zeros((cases, points))
    mask = jnp.asarray(
        [
            [True, True, False, True],
            [True, False, True, True],
            [False, True, True, True],
        ]
    )
    return phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(values=state, axes=(axis,)),
            "duration": phx.nn.operator.FunctionSamples(values=duration, axes=(axis,)),
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,), mask=mask)
        },
        case_axes=("case",),
    )


def _condition(batch, duration):
    values = jnp.broadcast_to(
        jnp.asarray(duration).reshape(batch.case_shape + (1,)),
        batch.case_shape + batch.input("duration").sample_shape,
    )
    return eqx.tree_at(
        lambda current: current.inputs["duration"].values,
        batch,
        values,
    )


def _advance(batch, prediction):
    return eqx.tree_at(
        lambda current: current.inputs["state"].values,
        batch,
        prediction,
    )


def test_exact_semigroup_has_zero_loss_with_batched_conditions():
    batch = _batch()
    dt1 = jnp.asarray([0.1, 0.2, 0.3])
    dt2 = jnp.asarray([0.4, 0.3, 0.2])
    objective = ConditionedSemigroupObjective()

    loss = objective(
        _ConditionedTransition(jnp.asarray(0.7), exact=True),
        batch,
        dt1,
        dt2,
        _condition,
        _advance,
    )

    assert loss.shape == ()
    assert jnp.allclose(loss, 0.0, atol=1e-12)


def test_violating_channel_transition_is_positive_and_respects_query_measure():
    batch = _batch(channels=2)
    model = _ConditionedTransition(jnp.asarray(0.8), exact=False)
    dt1 = jnp.asarray([0.2, 0.3, 0.4])
    dt2 = jnp.asarray([0.5, 0.4, 0.3])

    mean = conditioned_semigroup_consistency_loss(
        model,
        batch,
        dt1,
        dt2,
        _condition,
        _advance,
        reduction="mean",
    )
    summed = conditioned_semigroup_consistency_loss(
        model,
        batch,
        dt1,
        dt2,
        _condition,
        _advance,
        reduction="sum",
        weight=2.5,
    )
    query_mass = jnp.sum(
        batch.require_single_query().weights(case_shape=batch.case_shape)
    )
    channel_count = batch.input("state").values.shape[-1]

    assert mean > 0.0
    assert jnp.allclose(summed, 2.5 * mean * query_mass * channel_count)


def test_violating_semigroup_objective_backpropagates_to_transition_parameters():
    batch = _batch()
    model = _ConditionedTransition(jnp.asarray(0.6), exact=False)

    loss, gradient = eqx.filter_value_and_grad(
        lambda current: conditioned_semigroup_consistency_loss(
            current,
            batch,
            jnp.asarray([0.2, 0.25, 0.3]),
            jnp.asarray([0.4, 0.35, 0.3]),
            _condition,
            _advance,
        )
    )(model)

    assert loss > 0.0
    assert jnp.isfinite(gradient.rate)
    assert jnp.abs(gradient.rate) > 0.0


def test_semigroup_objective_rejects_non_case_condition_shapes():
    batch = _batch()

    with pytest.raises(ValueError, match="dt1 must be scalar or"):
        conditioned_semigroup_consistency_loss(
            _ConditionedTransition(jnp.asarray(0.6), exact=True),
            batch,
            jnp.ones((3, 1)),
            0.2,
            _condition,
            _advance,
        )


@pytest.mark.parametrize("key_mode", ["fold_in", "split"])
def test_semigroup_evaluation_key_modes_are_deterministic(key_mode):
    batch = _batch()
    objective = ConditionedSemigroupObjective(key_mode=key_mode)
    root = jr.key(23)

    first = objective(
        _KeyedTransition(),
        batch,
        jnp.asarray([0.1, 0.2, 0.3]),
        jnp.asarray([0.3, 0.2, 0.1]),
        _condition,
        _advance,
        key=root,
    )
    repeated = objective(
        _KeyedTransition(),
        batch,
        jnp.asarray([0.1, 0.2, 0.3]),
        jnp.asarray([0.3, 0.2, 0.1]),
        _condition,
        _advance,
        key=root,
    )

    assert jnp.array_equal(first, repeated)


def test_semigroup_public_exports_are_available_from_nn_namespace():
    assert (
        phx.nn.operator.training.ConditionedSemigroupObjective
        is ConditionedSemigroupObjective
    )
    assert (
        phx.nn.operator.training.conditioned_semigroup_consistency_loss
        is conditioned_semigroup_consistency_loss
    )
