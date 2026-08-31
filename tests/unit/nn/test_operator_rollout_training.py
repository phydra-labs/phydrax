#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


class _StateOperator(phx.nn.operator.AbstractOperatorModel):
    gain: jax.Array
    keyed: bool = eqx.field(static=True)
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, gain=1.0, *, keyed=False):
        self.gain = jnp.asarray(gain)
        self.keyed = bool(keyed)
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("FNO")

    def __call_operator_batch__(self, batch, *, key=None):
        state = batch.input("state").values
        assert state is not None
        values = self.gain * state
        if "control" in batch.inputs:
            control = batch.input("control").values
            assert control is not None
            values = values + control
        if self.keyed:
            values = values + jr.uniform(key, values.shape, dtype=values.dtype)
        return values

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _axis(*, shifted=False):
    nodes = jnp.linspace(0.1 if shifted else 0.0, 1.0, 4)
    return phx.nn.operator.OperatorAxis(
        "x",
        nodes,
        quadrature_weights=jnp.full((4,), 0.25),
    )


def _task(*, independent=False, control=False):
    fields = [
        phx.nn.operator.OperatorFieldSpec(
            "state",
            role="both",
            source_name="state",
            query_name="query",
        )
    ]
    if control:
        fields.append(
            phx.nn.operator.OperatorFieldSpec(
                "control",
                role="source",
                source_name="control",
            )
        )
    return phx.nn.operator.OperatorTask(
        "recurrent-state",
        fields=tuple(fields),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="independent" if independent else "coincident",
            query_is_fixed=False,
            rollout_steps=4,
        ),
    )


def _batch(values, *, control=None, mask=None, shifted_query=False):
    axis = _axis()
    inputs = {
        "state": phx.nn.operator.FunctionSamples(
            values=jnp.asarray(values),
            axes=(axis,),
            mask=mask,
        )
    }
    if control is not None:
        inputs["control"] = phx.nn.operator.FunctionSamples(
            values=jnp.asarray(control),
            axes=(axis,),
            mask=mask,
        )
    return phx.nn.operator.OperatorBatch(
        inputs=inputs,
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(_axis(shifted=shifted_query) if shifted_query else axis,),
                mask=mask,
            )
        },
        case_axes=("case",),
        case_shape=(int(jnp.asarray(values).shape[0]),),
    )


def _targets(batch, first, second):
    spec = phx.nn.operator.OperatorOutputSpec("scalar")
    return phx.nn.operator.OperatorTargetBatch(
        {
            "state_t1": phx.nn.operator.OperatorFieldBatch(
                first,
                query_name="query",
                spec=spec,
            ),
            "state_t2": phx.nn.operator.OperatorFieldBatch(
                second,
                query_name="query",
                spec=spec,
            ),
        },
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def _route():
    return phx.nn.operator.training.OperatorRolloutRoute(
        source_name="state",
        prediction_name="output",
        task_field="state",
    )


def _trained(model, task, *, normalization=None, output_pipeline=None):
    return phx.nn.operator.training.TrainedOperator(
        model,
        task,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "state"},
        normalization=normalization,
        output_pipeline=output_pipeline,
    )


def _dataset(*, cases=4, mask=None, target_nan=False):
    initial = jnp.arange(1, cases + 1, dtype=float)[:, None]
    values = jnp.broadcast_to(initial, (cases, 4))
    batch = _batch(values, mask=mask)
    first = 1.5 * values
    second = 2.25 * values
    if target_nan:
        first = first.at[:, -1].set(jnp.nan)
        second = second.at[:, -1].set(jnp.nan)
    return phx.nn.operator.training.OperatorDataset(
        batch,
        _targets(batch, first, second),
    )


def _supervised_loss():
    return phx.nn.operator.training.SupervisedOperatorRolloutLoss(
        target_fields=("state_t1", "state_t2"),
        time_weights=(1.0, 0.5),
    )


def _policy(**kwargs):
    return phx.nn.operator.training.OperatorRolloutPolicy(
        maximum_horizon=2,
        initial_horizon=2,
        **kwargs,
    )


def test_future_aliases_share_one_canonical_target_normalizer():
    dataset = _dataset(cases=2)
    aliases = {"state_t1": "state", "state_t2": "state"}
    policy = phx.nn.operator.training.fit_operator_normalization(
        dataset.batch,
        dataset.targets,
        fields=_task().fields,
        target_aliases=aliases,
    )

    assert tuple(policy.targets) == ("state",)
    normalized = policy.normalize_targets(
        dataset.targets,
        target_aliases=aliases,
    )
    restored = policy.denormalize_targets(normalized, target_aliases=aliases)
    assert jnp.allclose(
        restored.field("state_t1").values,
        dataset.targets.field("state_t1").values,
    )
    assert jnp.allclose(
        restored.field("state_t2").values,
        dataset.targets.field("state_t2").values,
    )


def test_feedback_reprepares_with_source_not_target_normalization():
    values = jnp.full((1, 4), 14.0)
    batch = _batch(values)
    normalizer = phx.nn.operator.training.AffineNormalizer
    policy = phx.nn.operator.training.OperatorNormalizationPolicy(
        input_values={
            "state": normalizer(
                mean=jnp.asarray(10.0),
                scale=jnp.asarray(2.0),
                channel_axis=None,
                epsilon=1e-6,
            )
        },
        targets={
            "state": normalizer(
                mean=jnp.asarray(100.0),
                scale=jnp.asarray(5.0),
                channel_axis=None,
                epsilon=1e-6,
            )
        },
        input_coordinates={},
        query_coordinates={},
    )
    rollout = phx.nn.operator.training.autoregressive_operator_rollout(
        _trained(_StateOperator(), _task(), normalization=policy),
        batch,
        2,
        _route(),
        key=jr.key(3),
    )

    assert jnp.allclose(rollout.predictions[0].field("state").values, 110.0)
    assert jnp.allclose(rollout.predictions[1].field("state").values, 350.0)


def _zero_envelope(coordinates, batch, *, key):
    del coordinates, batch, key
    return 0.0


def _state_plus_control(coordinates, batch, *, key):
    del coordinates, key
    state = batch.input("state").values
    control = batch.input("control").values
    assert state is not None and control is not None
    return state + control


def test_constrained_feedback_and_static_conditioning_recur_on_step_two():
    state = jnp.zeros((2, 4))
    control = jnp.stack((jnp.ones((4,)), jnp.full((4,), 2.0)))
    batch = _batch(state, control=control)
    pipeline = phx.nn.operator.training.OperatorOutputPipeline(
        phx.nn.operator.training.HardConstraintTransform(
            "state",
            _zero_envelope,
            "tests.zero-envelope",
            lift_fn=_state_plus_control,
        )
    )
    rollout = phx.nn.operator.training.autoregressive_operator_rollout(
        _trained(
            _StateOperator(),
            _task(control=True),
            output_pipeline=pipeline,
        ),
        batch,
        2,
        _route(),
    )

    first = rollout.predictions[0].field("state").values
    second = rollout.predictions[1].field("state").values
    assert jnp.array_equal(first, control)
    assert jnp.array_equal(second, 2.0 * control)
    assert jnp.array_equal(rollout.final_batch.input("control").values, control)


def test_route_rejects_independent_or_mismatched_support_and_multiple_routes():
    values = jnp.ones((1, 4))
    route = _route()
    independent = _trained(_StateOperator(), _task(independent=True))
    with pytest.raises(ValueError, match="coincident"):
        phx.nn.operator.training.autoregressive_operator_rollout(
            independent,
            _batch(values),
            1,
            route,
        )

    coincident = _trained(_StateOperator(), _task())
    with pytest.raises(ValueError, match="supports"):
        phx.nn.operator.training.autoregressive_operator_rollout(
            coincident,
            _batch(values, shifted_query=True),
            1,
            route,
        )
    bad_prediction = phx.nn.operator.training.OperatorRolloutRoute(
        source_name="state",
        prediction_name="missing",
        task_field="state",
    )
    with pytest.raises(ValueError, match="output map"):
        phx.nn.operator.training.autoregressive_operator_rollout(
            coincident,
            _batch(values),
            1,
            bad_prediction,
        )
    with pytest.raises(TypeError, match="OperatorRolloutRoute"):
        phx.nn.operator.training.autoregressive_operator_rollout(
            coincident,
            _batch(values),
            1,
            (route, route),
        )


def test_masked_future_nans_are_sanitized_before_rollout_residuals():
    mask = jnp.asarray([True, True, True, False])
    result = phx.nn.operator.training.fit_operator(
        _StateOperator(1.5),
        _dataset(mask=mask, target_nan=True),
        task=_task(),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "state"},
        loss_terms=(_supervised_loss(),),
        rollout_route=_route(),
        rollout_policy=_policy(),
        learning_rate=0.0,
        epochs=1,
        steps=1,
        batch_size=4,
        shuffle=False,
        jit=True,
    )

    assert jnp.isfinite(result.initial_loss)
    assert result.initial_loss == 0.0


def _prediction_energy(prediction, batch, targets, **kwargs):
    del batch, targets, kwargs
    return jnp.mean(prediction.field("state").values ** 2)


def test_residual_recurrence_has_gradients_and_honors_bptt_rematerialization():
    empty = phx.nn.operator.OperatorTargetBatch(
        {},
        case_axes=("case",),
        case_shape=(2,),
    )
    batch = _batch(jnp.ones((2, 4)))
    dataset = phx.nn.operator.training.OperatorDataset(batch, empty)
    residual = phx.nn.operator.training.ResidualOperatorRolloutLoss(
        residual_term=phx.nn.operator.training.OperatorLossTerm(
            "state_energy",
            _prediction_energy,
            identity="tests.state-energy",
        ),
        time_weights=(1.0, 1.0),
    )
    result = phx.nn.operator.training.fit_operator(
        _StateOperator(0.5),
        dataset,
        task=_task(),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "state"},
        loss_terms=(residual,),
        rollout_route=_route(),
        rollout_policy=_policy(truncate_every=1, rematerialize=True),
        learning_rate=0.1,
        epochs=1,
        steps=1,
        batch_size=2,
        shuffle=False,
    )

    assert result.execution_model.gain != 0.5
    schedule = phx.nn.operator.training.OperatorRolloutPolicy(
        maximum_horizon=4,
        initial_horizon=1,
        transition_steps=6,
    )
    assert jax.jit(schedule.active_horizon)(jnp.asarray(0)) == 1
    assert jax.jit(schedule.active_horizon)(jnp.asarray(3)) == 2
    assert jax.jit(schedule.active_horizon)(jnp.asarray(6)) == 4


def test_rollout_updates_are_batch_and_accumulation_invariant():
    dataset = _dataset(cases=4)
    common = {
        "task": _task(),
        "training_evidence": phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        "output_field_map": {"output": "state"},
        "loss_terms": (_supervised_loss(),),
        "rollout_route": _route(),
        "rollout_policy": _policy(),
        "learning_rate": 0.01,
        "epochs": 1,
        "steps": 1,
        "shuffle": False,
        "seed": 11,
    }
    full = phx.nn.operator.training.fit_operator(
        _StateOperator(1.0),
        dataset,
        batch_size=4,
        **common,
    )
    accumulated = phx.nn.operator.training.fit_operator(
        _StateOperator(1.0),
        dataset,
        batch_size=2,
        gradient_accumulation=2,
        **common,
    )

    assert jnp.allclose(
        full.last_execution_model.gain,
        accumulated.last_execution_model.gain,
    )
    assert jnp.allclose(
        full.history.train_metrics[0]["loss"],
        accumulated.history.train_metrics[0]["loss"],
    )


def test_full_prefix_and_chunk_rollouts_share_semantic_step_keys():
    batch = _batch(jnp.zeros((1, 4)))
    trained = _trained(_StateOperator(keyed=True), _task())
    key = jr.key(19)
    full = phx.nn.operator.training.autoregressive_operator_rollout(
        trained,
        batch,
        4,
        _route(),
        key=key,
    )
    prefix = phx.nn.operator.training.autoregressive_operator_rollout(
        trained,
        batch,
        2,
        _route(),
        key=key,
    )
    suffix = phx.nn.operator.training.autoregressive_operator_rollout(
        trained,
        prefix.final_batch,
        2,
        _route(),
        key=key,
        step_offset=prefix.next_step,
    )

    chunked = prefix.predictions + suffix.predictions
    for expected, actual in zip(full.predictions, chunked, strict=True):
        assert jnp.array_equal(
            expected.field("state").values,
            actual.field("state").values,
        )
    assert prefix.next_step == 2
    assert suffix.next_step == full.next_step == 4


def test_fit_and_deployment_use_the_same_recurrent_physical_pipeline():
    dataset = _dataset(cases=2)
    result = phx.nn.operator.training.fit_operator(
        _StateOperator(1.5),
        dataset,
        task=_task(),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "state"},
        loss_terms=(_supervised_loss(),),
        rollout_route=_route(),
        rollout_policy=_policy(),
        learning_rate=0.0,
        epochs=1,
        steps=0,
        batch_size=2,
        shuffle=False,
    )
    assert result.trained_operator is not None
    deployed = phx.nn.operator.training.autoregressive_operator_rollout(
        result.trained_operator,
        dataset.batch,
        2,
        _route(),
        key=jr.key(0),
    )

    assert result.initial_loss == 0.0
    assert jnp.array_equal(
        deployed.predictions[0].field("state").values,
        dataset.targets.field("state_t1").values,
    )
    assert jnp.array_equal(
        deployed.predictions[1].field("state").values,
        dataset.targets.field("state_t2").values,
    )
