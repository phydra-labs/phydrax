import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.layers import (
    GRUCell,
    LSTMCell,
    RecurrentBatch,
    RNNCell,
    StackedRecurrentCell,
)
from phydrax.nn.models import (
    BidirectionalRecurrentSequenceModel,
    RecurrentSequenceModel,
)


@pytest.mark.parametrize("cell_type", (GRUCell, LSTMCell))
def test_recurrent_cell_adapters_match_equinox_single_step_equations(cell_type):
    cell = cell_type(3, 5, dtype=jnp.float64, key=jr.key(1))
    inputs = jr.normal(jr.key(2), (3,))
    state = cell.initial_state((), dtype=jnp.float64)

    next_state, output = cell.step(state, inputs)
    expected = cell.cell(inputs, state)
    expected_output = expected[0] if isinstance(expected, tuple) else expected
    assert all(
        jnp.allclose(actual, reference)
        for actual, reference in zip(
            jax.tree.leaves(next_state),
            jax.tree.leaves(expected),
            strict=True,
        )
    )
    assert jnp.allclose(output, expected_output)


def test_rnn_cell_matches_its_declared_elman_equation():
    cell = RNNCell(2, 3, activation="tanh", dtype=jnp.float64, key=jr.key(3))
    inputs = jnp.array([0.2, -0.4])
    state = jnp.array([0.5, 0.1, -0.3])
    next_state, output = cell.step(state, inputs)
    expected = jnp.tanh(cell.weight_ih @ inputs + cell.weight_hh @ state + cell.bias)
    assert jnp.allclose(next_state, expected)
    assert jnp.array_equal(output, next_state)


def test_stacked_recurrent_model_supports_nested_lstm_state_and_unequal_widths():
    stack = StackedRecurrentCell(
        (
            LSTMCell(2, 4, key=jr.key(4)),
            GRUCell(4, 3, key=jr.key(5)),
        )
    )
    model = RecurrentSequenceModel(stack)
    valid = jnp.array([[True, True, True, False], [True, True, True, True]])
    batch = RecurrentBatch(jnp.ones((2, 4, 2)), valid)

    result = model.evaluate_with_state(batch)
    output = eqx.filter_jit(lambda current: current(batch))(model)
    assert output.shape == (2, 4, 3)
    assert jnp.array_equal(output[0, 3], jnp.zeros((3,)))
    assert isinstance(result.final_state, tuple)
    assert isinstance(result.final_state[0], tuple)
    assert result.final_state[0][0].shape == (2, 4)
    assert result.final_state[1].shape == (2, 3)


def test_recurrent_sequence_readout_cannot_reintroduce_values_on_padding():
    cell = GRUCell(2, 3, key=jr.key(6))
    biased_readout = eqx.nn.Linear(3, 2, key=jr.key(7))
    model = RecurrentSequenceModel(cell, readout=biased_readout)
    valid = jnp.array([True, True, False, False])
    output = model(RecurrentBatch(jnp.ones((4, 2)), valid))
    assert output.shape == (4, 2)
    assert jnp.array_equal(output[2:], jnp.zeros((2, 2)))


def test_bidirectional_recurrence_reverses_each_reset_delimited_segment_independently():
    forward = GRUCell(2, 3, dtype=jnp.float64, key=jr.key(8))
    backward = GRUCell(2, 3, dtype=jnp.float64, key=jr.key(9))
    model = BidirectionalRecurrentSequenceModel(forward, backward)
    inputs = jr.normal(jr.key(10), (6, 2))
    valid = jnp.ones((6,), dtype=bool)
    reset = jnp.array([False, False, False, True, False, False])

    packed = model(RecurrentBatch(inputs, valid, reset=reset))
    first = model(RecurrentBatch(inputs[:3], jnp.ones((3,), dtype=bool)))
    second = model(RecurrentBatch(inputs[3:], jnp.ones((3,), dtype=bool)))
    assert jnp.allclose(packed, jnp.concatenate((first, second)), atol=1e-10, rtol=1e-10)


def test_recurrent_models_are_vmappable_differentiable_and_support_final_readout():
    cell = GRUCell(2, 4, dtype=jnp.float64, key=jr.key(11))
    sequence_model = RecurrentSequenceModel(cell)
    final_model = RecurrentSequenceModel(cell, return_mode="final")
    valid = jnp.array([True, True, True, False])
    inputs = jr.normal(jr.key(12), (3, 4, 2))

    vmapped = jax.vmap(lambda values: sequence_model(RecurrentBatch(values, valid)))(
        inputs
    )
    assert vmapped.shape == (3, 4, 4)
    assert final_model(RecurrentBatch(inputs[0], valid)).shape == (4,)
    gradient = jax.grad(
        lambda values: jnp.sum(sequence_model(RecurrentBatch(values, valid)) ** 2)
    )(inputs[0])
    assert jnp.all(jnp.isfinite(gradient))
