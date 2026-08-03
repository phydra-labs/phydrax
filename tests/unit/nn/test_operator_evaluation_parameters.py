#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx
from phydrax._trainable import combine_trainable, partition_trainable


def _dataset():
    axis = phx.nn.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, 4),
        quadrature_weights=jnp.full((4,), 0.25),
    )
    values = jnp.stack((axis.nodes, axis.nodes + 1.0))
    return phx.nn.operator_dataset_from_arrays(
        {"state": values},
        {"solution": 2.0 * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )


def _model():
    return phx.nn.FNO(
        in_channels="scalar",
        out_channels="scalar",
        width=2,
        depth=1,
        n_modes=(2,),
        key=jr.key(0),
    )


def _schedule_free():
    return optax.contrib.schedule_free(optax.sgd(1e-3), 1e-3)


def _schedule_free_fit(model, dataset, **kwargs):
    return phx.nn.fit_operator(
        model,
        dataset,
        optimizer=_schedule_free(),
        optimizer_id="tests.schedule_free_sgd.v1",
        evaluation_parameters=optax.contrib.schedule_free_eval_params,
        evaluation_parameters_id="optax.schedule_free_eval_params.v1",
        batch_size=2,
        seed=11,
        jit=False,
        **kwargs,
    )


def _assert_array_trees_equal(left, right):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if isinstance(left_leaf, jax.Array):
            assert jnp.array_equal(left_leaf, right_leaf)


def test_fit_operator_returns_and_validates_on_evaluation_model():
    dataset = _dataset()
    model = _model()

    def shifted(_state, parameters):
        return jax.tree.map(lambda value: value + 1.0, parameters)

    baseline = phx.nn.fit_operator(
        model,
        dataset,
        validation=dataset,
        steps=0,
        jit=False,
    )
    evaluated = phx.nn.fit_operator(
        model,
        dataset,
        validation=dataset,
        steps=0,
        evaluation_parameters=shifted,
        jit=False,
    )
    parameters, fixed = partition_trainable(baseline.last_execution_model)
    expected = combine_trainable(
        jax.tree.map(lambda value: value + 1.0, parameters),
        fixed,
    )

    assert bool(eqx.tree_equal(evaluated.last_execution_model, expected))
    assert evaluated.history.validation_metrics[-1]["loss"] == evaluated.final_loss
    assert evaluated.final_loss != baseline.final_loss


def test_schedule_free_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    dataset = _dataset()
    model = _model()
    uninterrupted = _schedule_free_fit(
        model,
        dataset,
        epochs=2,
        steps=2,
    )
    checkpoint = tmp_path / "schedule-free-fit"
    _schedule_free_fit(
        model,
        dataset,
        epochs=1,
        steps=1,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
    )
    resumed = _schedule_free_fit(
        model,
        dataset,
        epochs=2,
        steps=2,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        resume=True,
    )

    _assert_array_trees_equal(
        uninterrupted.last_execution_model,
        resumed.last_execution_model,
    )
    assert resumed.resumed_from_step == 1
    assert resumed.history == uninterrupted.history


@pytest.mark.parametrize("resume_id", ["changed-evaluator", None])
def test_schedule_free_resume_rejects_changed_or_missing_evaluator_id(
    tmp_path,
    resume_id,
):
    dataset = _dataset()
    model = _model()
    checkpoint = tmp_path / "schedule-free-contract"
    _schedule_free_fit(
        model,
        dataset,
        epochs=1,
        steps=1,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
    )

    kwargs = {
        "optimizer": _schedule_free(),
        "optimizer_id": "tests.schedule_free_sgd.v1",
        "batch_size": 2,
        "seed": 11,
        "jit": False,
        "epochs": 2,
        "steps": 2,
        "checkpoint_path": checkpoint,
        "checkpoint_every": 1,
        "resume": True,
    }
    if resume_id is not None:
        kwargs["evaluation_parameters"] = optax.contrib.schedule_free_eval_params
        kwargs["evaluation_parameters_id"] = resume_id

    with pytest.raises(ValueError, match="checkpoint contract mismatch"):
        phx.nn.fit_operator(model, dataset, **kwargs)


def test_checkpointed_evaluation_transform_requires_stable_id(tmp_path):
    with pytest.raises(ValueError, match="stable evaluation_parameters_id"):
        phx.nn.fit_operator(
            _model(),
            _dataset(),
            steps=0,
            checkpoint_path=tmp_path / "fit",
            evaluation_parameters=optax.contrib.schedule_free_eval_params,
            optimizer=_schedule_free(),
            optimizer_id="tests.schedule_free_sgd.v1",
            jit=False,
        )


def test_operator_evaluation_transform_must_preserve_parameter_structure():
    with pytest.raises(ValueError, match="PyTree structure"):
        phx.nn.fit_operator(
            _model(),
            _dataset(),
            steps=0,
            evaluation_parameters=lambda _state, parameters: (parameters,),
            jit=False,
        )
