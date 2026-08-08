import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


class _ScalarRateModel(eqx.Module):
    rate: jax.Array

    def __call__(self, state):
        return jnp.asarray([self.rate * state[0]])


def _time_path(path_id):
    return phx.solver.CallableDrivingPath(
        lambda time, side: jnp.asarray([time]),
        lambda time, side: jnp.asarray([1.0]),
        support=jnp.asarray([0.0, 1.0]),
        value_shape=(1,),
        path_id=path_id,
        breakpoints=jnp.empty((0,)),
        breakpoint_mask=jnp.empty((0,), dtype=bool),
    )


def _training_data():
    rate = 0.4
    initial = jnp.asarray([[1.0], [1.5], [0.7]])
    times = jnp.asarray(
        [
            [0.15, 0.55, 0.95],
            [0.10, 0.80, jnp.nan],
            [0.20, 0.45, 1.00],
        ]
    )
    valid = jnp.asarray(
        [
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ]
    )
    safe_times = jnp.where(valid, times, 0.0)
    targets = initial[:, None, :] * jnp.exp(rate * safe_times[..., None])
    targets = jnp.where(valid[..., None], targets, jnp.nan)
    return phx.solver.NeuralCDETrainingData(
        tuple(_time_path(f"explicit-time:{index}") for index in range(3)),
        initial,
        times,
        targets,
        valid=valid,
        time_channel=0,
        case_ids=("alpha", "beta", "gamma"),
    )


def test_vector_field_adapter_composes_callable_mlp_and_kan_models():
    callable_field = phx.solver.NeuralCDEVectorField(
        lambda state: jnp.asarray([state[0], -state[0]]),
        state_shape=(1,),
        control_dimension=2,
    )
    mlp_field = phx.solver.NeuralCDEVectorField(
        phx.nn.MLP(
            in_size=1,
            out_size=2,
            width_size=3,
            depth=1,
            key=jr.key(1),
        ),
        state_shape=(1,),
        control_dimension=2,
    )
    kan_field = phx.solver.NeuralCDEVectorField(
        phx.nn.KAN(
            in_size=1,
            out_size=2,
            width_size=3,
            depth=1,
            key=jr.key(2),
        ),
        state_shape=(1,),
        control_dimension=2,
    )

    state = jnp.asarray([0.25])
    for field in (callable_field, mlp_field, kan_field):
        value = field(jnp.asarray(0.3), state, None)
        assert value.shape == (1, 2)
        assert jnp.all(jnp.isfinite(value))


def test_irregular_observation_loss_uses_mask_and_explicit_time_channel():
    data = _training_data()
    vector_field = phx.solver.NeuralCDEVectorField(
        _ScalarRateModel(jnp.asarray(0.4)),
        state_shape=(1,),
        control_dimension=1,
    )

    loss = phx.solver.neural_cde_loss(
        vector_field,
        data,
        solve_options={"rtol": 1e-9, "atol": 1e-11},
    )

    assert data.time_channel == 0
    assert data.observation_indices == ((0, 1, 2), (0, 1), (0, 1, 2))
    assert data.case_ids == ("alpha", "beta", "gamma")
    assert data.data_id
    assert jnp.isfinite(loss)
    assert loss < 1e-13


def test_complex_residual_uses_hermitian_squared_norm():
    data = phx.solver.NeuralCDETrainingData(
        (_time_path("complex-residual"),),
        jnp.asarray([[1.0j]]),
        jnp.asarray([[0.5]]),
        jnp.asarray([[[0.0j]]]),
        time_channel=0,
    )

    loss = phx.solver.neural_cde_loss(
        lambda time, state, args: jnp.zeros(
            state.shape + (1,),
            dtype=state.dtype,
        ),
        data,
    )

    assert jnp.isrealobj(loss)
    assert jnp.allclose(loss, 1.0)


def test_failed_requested_save_rejects_loss_before_optimizer_update():
    data = _training_data()
    field = phx.solver.NeuralCDEVectorField(
        _ScalarRateModel(jnp.asarray(0.1)),
        state_shape=(1,),
        control_dimension=1,
    )
    failure_options = {"dt0": 1e-6, "max_steps": 1}

    with pytest.raises(Exception, match="did not produce every requested save"):
        loss = phx.solver.neural_cde_loss(
            field,
            data,
            indices=(0,),
            solve_options=failure_options,
        )
        jax.block_until_ready(loss)

    update_calls = []

    def init_optimizer(parameters):
        del parameters
        return ()

    def update_optimizer(gradients, state, parameters=None):
        del parameters
        update_calls.append(True)
        return gradients, state

    optimizer = optax.GradientTransformation(init_optimizer, update_optimizer)
    original_rate = field.model.rate
    with pytest.raises(Exception, match="did not produce every requested save"):
        phx.solver.train_neural_cde(
            data,
            optimizer=optimizer,
            num_steps=1,
            batch_size=1,
            vector_field=field,
            optimizer_id="must-not-update",
            shuffle=False,
            solve_options=failure_options,
        )

    assert update_calls == []
    assert jnp.array_equal(field.model.rate, original_rate)


def test_optax_training_resume_is_step_for_step_equivalent():
    data = _training_data()
    optimizer = optax.adam(2e-2)
    solve_options = {"rtol": 2e-6, "atol": 2e-8}

    uninterrupted = phx.solver.train_neural_cde(
        data,
        optimizer=optimizer,
        num_steps=5,
        batch_size=2,
        vector_field=phx.solver.NeuralCDEVectorField(
            _ScalarRateModel(jnp.asarray(0.1)),
            state_shape=(1,),
            control_dimension=1,
        ),
        optimizer_id="adam:2e-2",
        seed=13,
        shuffle=True,
        solve_options=solve_options,
    )
    partial = phx.solver.train_neural_cde(
        data,
        optimizer=optimizer,
        num_steps=2,
        batch_size=2,
        vector_field=phx.solver.NeuralCDEVectorField(
            _ScalarRateModel(jnp.asarray(0.1)),
            state_shape=(1,),
            control_dimension=1,
        ),
        optimizer_id="adam:2e-2",
        seed=13,
        shuffle=True,
        solve_options=solve_options,
    )
    resumed = phx.solver.train_neural_cde(
        data,
        optimizer=optimizer,
        num_steps=3,
        batch_size=2,
        state=partial,
        optimizer_id="adam:2e-2",
        seed=13,
        shuffle=True,
        solve_options=solve_options,
    )

    assert uninterrupted.update_step == resumed.update_step == 5
    assert uninterrupted.epoch == resumed.epoch
    assert uninterrupted.batch_index == resumed.batch_index
    assert uninterrupted.training_id == resumed.training_id
    assert uninterrupted.data_id == data.data_id
    assert uninterrupted.ordering == resumed.ordering
    assert jnp.array_equal(
        uninterrupted.vector_field.model.rate,
        resumed.vector_field.model.rate,
    )
    assert jnp.array_equal(uninterrupted.last_loss, resumed.last_loss)
    uninterrupted_optimizer = jax.tree.leaves(uninterrupted.optimizer_state)
    resumed_optimizer = jax.tree.leaves(resumed.optimizer_state)
    assert len(uninterrupted_optimizer) == len(resumed_optimizer)
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(uninterrupted_optimizer, resumed_optimizer, strict=True)
    )
