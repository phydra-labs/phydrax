import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel, ModelBinding
from phydrax.dynamics.identification._neural import _objective_contributions
from phydrax.dynamics.identification._neural_windows import _NeuralWindowSource


class _ScaledStep(AbstractArrayModel):
    scale: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale):
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, state, /, *, key=None):
        del key
        return self.scale * state


class _ControlledStep(AbstractArrayModel):
    scale: jax.Array
    in_size: tuple[int, int] = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    _input_binding = ModelBinding.pointwise("structured")

    def __init__(self, scale=1.0):
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        self.in_size = (1, 1)
        self.out_size = 1

    def __call__(self, values, /, *, key=None):
        del key
        state, control = values
        return self.scale * state + control


class _KeyedStep(AbstractArrayModel):
    scale: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale):
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, state, /, *, key=None):
        return self.scale * state + 0.01 * jr.normal(key, state.shape)


class _DropoutStep(AbstractArrayModel):
    scale: jax.Array
    dropout: eqx.nn.Dropout
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale):
        self.scale = jnp.asarray(scale, dtype=jnp.float32)
        self.dropout = eqx.nn.Dropout(p=0.5)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, state, /, *, key=None):
        return self.dropout(self.scale * state, key=key)


def _trajectory(
    states,
    *,
    coordinates=None,
    sample_valid=None,
    transition_valid=None,
    reset_mask=None,
    weights=None,
    inputs=None,
    input_alignment="transitions",
):
    values = jnp.asarray(states, dtype=jnp.float32).reshape((-1, 1))
    count = values.shape[0]
    return phx.dynamics.TrajectoryData(
        (
            jnp.arange(count, dtype=jnp.float32)
            if coordinates is None
            else jnp.asarray(coordinates, dtype=jnp.float32)
        ),
        values,
        state_layout=phx.dynamics.StateLayout((1,)),
        sample_valid=sample_valid,
        transition_valid=transition_valid,
        reset_mask=reset_mask,
        weights=weights,
        inputs=inputs,
        input_layout=(None if inputs is None else phx.dynamics.InputLayout((1,))),
        input_alignment=input_alignment,
        source_id="neural-test",
    )


def _source(data, horizon):
    return _NeuralWindowSource(
        data,
        max_horizon=horizon,
        step_size=1.0,
        step_rtol=0.0,
        step_atol=0.0,
    )


def _loss(model, batch, policy, objectives=None):
    terms = (
        (phx.dynamics.identification.SupervisedDiscreteModelObjective(),)
        if objectives is None
        else tuple(objectives)
    )
    contribution, components, valid = _objective_contributions(
        model,
        batch,
        policy.active_horizon(jnp.asarray(0)),
        policy,
        terms,
        phx.dynamics.StateLayout((1,)),
        jr.key(17),
    )
    assert bool(valid)
    return contribution.value, components


def test_supervised_rollout_matches_manual_value_gradient_and_truncation():
    data = _trajectory([1.0, 3.0, 7.0])
    batch = _source(data, 2).prepare(jnp.arange(2))
    full = phx.dynamics.identification.DiscreteModelRolloutPolicy(max_horizon=2)
    truncated = phx.dynamics.identification.DiscreteModelRolloutPolicy(
        max_horizon=2,
        truncate_every=1,
    )

    value, gradient = eqx.filter_value_and_grad(
        lambda model: _loss(model, batch, full)[0]
    )(_ScaledStep(2.0))
    truncated_value, truncated_gradient = eqx.filter_value_and_grad(
        lambda model: _loss(model, batch, truncated)[0]
    )(_ScaledStep(2.0))

    np.testing.assert_allclose(value, 5.0)
    np.testing.assert_allclose(gradient.scale, -13.0)
    np.testing.assert_allclose(truncated_value, value)
    np.testing.assert_allclose(truncated_gradient.scale, -7.0)


def test_rematerialization_and_semantic_chunk_keys_preserve_value_and_gradient():
    data = _trajectory([1.0, 1.5, 2.0, 2.5, 3.0])
    source = _source(data, 2)
    full_batch = source.prepare(jnp.arange(source.size))
    plain = phx.dynamics.identification.DiscreteModelRolloutPolicy(max_horizon=2)
    rematerialized = phx.dynamics.identification.DiscreteModelRolloutPolicy(
        max_horizon=2,
        rematerialize=True,
    )
    model = _KeyedStep(1.1)

    plain_pair = eqx.filter_value_and_grad(
        lambda candidate: _loss(candidate, full_batch, plain)[0]
    )(model)
    rematerialized_pair = eqx.filter_value_and_grad(
        lambda candidate: _loss(candidate, full_batch, rematerialized)[0]
    )(model)
    np.testing.assert_allclose(rematerialized_pair[0], plain_pair[0])
    np.testing.assert_allclose(rematerialized_pair[1].scale, plain_pair[1].scale)

    left = source.prepare(jnp.arange(2))
    right = source.prepare(jnp.arange(2, source.size))
    left_contribution = _objective_contributions(
        model,
        left,
        jnp.asarray(2),
        plain,
        (phx.dynamics.identification.SupervisedDiscreteModelObjective(),),
        data.state_layout,
        jr.key(17),
    )[0]
    right_contribution = _objective_contributions(
        model,
        right,
        jnp.asarray(2),
        plain,
        (phx.dynamics.identification.SupervisedDiscreteModelObjective(),),
        data.state_layout,
        jr.key(17),
    )[0]
    np.testing.assert_allclose(
        left_contribution.numerator + right_contribution.numerator,
        plain_pair[0] * (left_contribution.support + right_contribution.support),
    )


def test_lazy_windows_align_both_control_conventions_and_endpoint_evidence():
    transition_data = _trajectory(
        [0.0, 1.0, 3.0, 6.0],
        inputs=jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float32),
        input_alignment="transitions",
        weights=jnp.asarray([4.0, 1.0, 9.0, 16.0]),
    )
    sample_data = _trajectory(
        [0.0, 1.0, 3.0, 6.0],
        inputs=jnp.asarray([[1.0], [2.0], [3.0], [99.0]], dtype=jnp.float32),
        input_alignment="samples",
        weights=jnp.asarray([4.0, 1.0, 9.0, 16.0]),
    )

    transition_batch = _source(transition_data, 2).prepare(jnp.asarray([0]))
    sample_batch = _source(sample_data, 2).prepare(jnp.asarray([0]))

    np.testing.assert_array_equal(transition_batch.inputs, sample_batch.inputs)
    np.testing.assert_array_equal(
        transition_batch.inputs[0, :, 0],
        jnp.asarray([1.0, 2.0]),
    )
    np.testing.assert_allclose(
        jnp.sqrt(transition_batch.weights[0, 0] * transition_batch.weights[0, 2]),
        6.0,
    )
    policy = phx.dynamics.identification.DiscreteModelRolloutPolicy(max_horizon=2)
    np.testing.assert_allclose(
        _loss(_ControlledStep(), transition_batch, policy)[0],
        0.0,
    )
    np.testing.assert_allclose(
        _loss(_ControlledStep(), sample_batch, policy)[0],
        0.0,
    )


def test_invalid_reset_nan_padding_is_sanitized_and_has_zero_support():
    data = _trajectory(
        [1.0, 2.0, jnp.nan, jnp.nan],
        coordinates=[0.0, 1.0, jnp.nan, jnp.nan],
        sample_valid=jnp.asarray([True, True, False, False]),
        transition_valid=jnp.asarray([True, False, False]),
        reset_mask=jnp.asarray([False, True, False]),
    )
    source = _source(data, 2)
    batch = source.prepare(jnp.arange(source.size))

    assert bool(jnp.all(jnp.isfinite(batch.coordinates)))
    assert bool(jnp.all(jnp.isfinite(batch.states)))
    value, components = _loss(
        _ScaledStep(2.0),
        batch,
        phx.dynamics.identification.DiscreteModelRolloutPolicy(max_horizon=2),
    )
    assert float(value) == 0.0
    assert float(components[0].support) == 0.0
    result = phx.dynamics.identification.fit_discrete_model(
        _ScaledStep(2.0),
        data,
        state_layout=data.state_layout,
        system_id="zero-support",
        step_size=1.0,
        rollout_policy=phx.dynamics.identification.DiscreteModelRolloutPolicy(
            max_horizon=2
        ),
        steps=1,
        gradient_accumulation=2,
        shuffle=False,
    )
    assert result.completed_steps == 0
    np.testing.assert_array_equal(result.last_model.scale, jnp.asarray(2.0))


def test_reference_branch_and_residual_objectives_match_manual_values():
    data = _trajectory([1.0, 2.0, 4.0])
    batch = _source(data, 2).prepare(jnp.arange(2))
    policy = phx.dynamics.identification.DiscreteModelRolloutPolicy(max_horizon=2)
    reference = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: 2.0 * state,
        state_layout=data.state_layout,
        system_id="doubling-reference",
        step_size=1.0,
        step_rtol=0.0,
        step_atol=0.0,
    )
    reference_term = phx.dynamics.identification.ReferenceBranchDiscreteModelObjective(
        reference,
        branch_length=1,
    )
    residual_term = phx.dynamics.identification.ResidualDiscreteModelObjective(
        lambda next_state, previous_state, coordinate, inputs: (
            next_state - 2.0 * previous_state
        ),
        residual_id="doubling-residual",
    )

    reference_value = _loss(_ScaledStep(1.5), batch, policy, (reference_term,))[0]
    residual_value = _loss(_ScaledStep(1.5), batch, policy, (residual_term,))[0]

    assert float(reference_value) > 0.0
    np.testing.assert_allclose(reference_value, residual_value)
    stopped_term = phx.dynamics.identification.ReferenceBranchDiscreteModelObjective(
        reference,
        branch_length=1,
        reference_gradient="stopped",
    )
    coupled_gradient = eqx.filter_grad(
        lambda model: _loss(model, batch, policy, (reference_term,))[0]
    )(_ScaledStep(1.5))
    stopped_gradient = eqx.filter_grad(
        lambda model: _loss(model, batch, policy, (stopped_term,))[0]
    )(_ScaledStep(1.5))
    assert not jnp.allclose(coupled_gradient.scale, stopped_gradient.scale)


def test_full_reference_branch_equals_reference_generated_supervision():
    data = _trajectory([1.0, 2.0, 4.0])
    batch = _source(data, 2).prepare(jnp.arange(2))
    policy = phx.dynamics.identification.DiscreteModelRolloutPolicy(max_horizon=2)
    reference = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: 2.0 * state,
        state_layout=data.state_layout,
        system_id="doubling-reference-full",
        step_size=1.0,
        step_rtol=0.0,
        step_atol=0.0,
    )
    reference_term = phx.dynamics.identification.ReferenceBranchDiscreteModelObjective(
        reference,
        branch_length=2,
    )
    supervised_term = phx.dynamics.identification.SupervisedDiscreteModelObjective()
    model = _ScaledStep(1.5)

    reference_pair = eqx.filter_value_and_grad(
        lambda candidate: _loss(candidate, batch, policy, (reference_term,))[0]
    )(model)
    supervised_pair = eqx.filter_value_and_grad(
        lambda candidate: _loss(candidate, batch, policy, (supervised_term,))[0]
    )(model)

    np.testing.assert_allclose(reference_pair[0], supervised_pair[0])
    np.testing.assert_allclose(reference_pair[1].scale, supervised_pair[1].scale)


def test_fixed_step_data_rejection_and_batch_accumulation_invariance():
    bad = _trajectory([1.0, 2.0, 3.0], coordinates=[0.0, 1.0, 2.5])
    with pytest.raises(ValueError, match="step_size"):
        phx.dynamics.identification.fit_discrete_model(
            _ScaledStep(0.5),
            bad,
            state_layout=bad.state_layout,
            system_id="bad-spacing",
            step_size=1.0,
            rollout_policy=phx.dynamics.identification.DiscreteModelRolloutPolicy(
                max_horizon=1
            ),
            steps=1,
            shuffle=False,
        )

    data = _trajectory(
        [1.0, 2.0, 4.0, 8.0, 16.0],
        weights=[1.0, 4.0, 9.0, 16.0, 25.0],
    )
    kwargs = {
        "state_layout": data.state_layout,
        "system_id": "batch-invariance",
        "step_size": 1.0,
        "rollout_policy": (
            phx.dynamics.identification.DiscreteModelRolloutPolicy(max_horizon=1)
        ),
        "steps": 1,
        "epochs": 1,
        "shuffle": False,
    }
    full = phx.dynamics.identification.fit_discrete_model(
        _ScaledStep(1.5),
        data,
        batch_size=4,
        gradient_accumulation=1,
        **kwargs,
    )
    accumulated = phx.dynamics.identification.fit_discrete_model(
        _ScaledStep(1.5),
        data,
        batch_size=3,
        gradient_accumulation=2,
        **kwargs,
    )
    np.testing.assert_allclose(full.last_model.scale, accumulated.last_model.scale)


def test_checkpoint_resume_is_exact_and_rejects_objective_mismatch(tmp_path):
    data = _trajectory([1.0, 2.0, 4.0, 8.0])
    policy = phx.dynamics.identification.DiscreteModelRolloutPolicy(max_horizon=1)
    common = {
        "state_layout": data.state_layout,
        "system_id": "resume-map",
        "model_id": "tests.scaled-step",
        "step_size": 1.0,
        "rollout_policy": policy,
        "batch_size": 2,
        "gradient_accumulation": 2,
        "shuffle": False,
    }
    uninterrupted = phx.dynamics.identification.fit_discrete_model(
        _ScaledStep(1.0),
        data,
        epochs=2,
        steps=2,
        **common,
    )
    checkpoint = tmp_path / "neural-checkpoint"
    phx.dynamics.identification.fit_discrete_model(
        _ScaledStep(1.0),
        data,
        epochs=1,
        steps=1,
        checkpoint_path=checkpoint,
        **common,
    )
    resumed = phx.dynamics.identification.fit_discrete_model(
        _ScaledStep(1.0),
        data,
        epochs=2,
        steps=2,
        checkpoint_path=checkpoint,
        resume=True,
        **common,
    )
    np.testing.assert_array_equal(
        uninterrupted.last_model.scale, resumed.last_model.scale
    )
    assert resumed.resumed_from_step == 1

    with pytest.raises(ValueError, match="fit contract mismatch"):
        phx.dynamics.identification.fit_discrete_model(
            _ScaledStep(1.0),
            data,
            epochs=2,
            steps=2,
            objectives=(
                phx.dynamics.identification.ResidualDiscreteModelObjective(
                    lambda next_state, previous_state, coordinate, inputs: (
                        next_state - previous_state
                    ),
                    residual_id="changed-residual",
                ),
            ),
            checkpoint_path=checkpoint,
            resume=True,
            **common,
        )


def test_rollout_coefficients_require_positive_mass_at_every_reachable_horizon():
    data = _trajectory([1.0, 2.0, 4.0])
    policy = phx.dynamics.identification.DiscreteModelRolloutPolicy(
        max_horizon=2,
        min_horizon=1,
        transition_steps=2,
        schedule="linear",
    )

    with pytest.raises(ValueError, match="positive mass"):
        phx.dynamics.identification.fit_discrete_model(
            _ScaledStep(1.0),
            data,
            state_layout=data.state_layout,
            system_id="zero-prefix-mass",
            step_size=1.0,
            rollout_policy=policy,
            objectives=(
                phx.dynamics.identification.SupervisedDiscreteModelObjective(
                    time_weights=(0.0, 1.0),
                ),
            ),
            steps=0,
        )


def test_fit_rejects_key_required_deployment_and_freezes_dropout_inference():
    data = _trajectory([1.0, 2.0, 4.0])
    policy = phx.dynamics.identification.DiscreteModelRolloutPolicy(max_horizon=1)
    common = {
        "state_layout": data.state_layout,
        "system_id": "keyless-deployment",
        "step_size": 1.0,
        "rollout_policy": policy,
        "steps": 0,
    }

    with pytest.raises(TypeError):
        phx.dynamics.identification.fit_discrete_model(
            _KeyedStep(1.0),
            data,
            **common,
        )

    fitted = phx.dynamics.identification.fit_discrete_model(
        _DropoutStep(2.0),
        data,
        **common,
    )
    state = jnp.asarray([3.0], dtype=jnp.float32)
    np.testing.assert_allclose(fitted.system.evaluate(0.0, state, None), 2.0 * state)
