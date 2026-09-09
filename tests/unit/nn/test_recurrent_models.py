import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.layers import (
    AbstractTimeAwareRecurrentCell,
    CfCCell,
    GRUCell,
    LSTMCell,
    RecurrentBatch,
    RNNCell,
    run_recurrent,
    StackedRecurrentCell,
)
from phydrax.nn.models import (
    BidirectionalRecurrentSequenceModel,
    RecurrentSequenceModel,
)


class _PhysicalVectorAccumulator(AbstractTimeAwareRecurrentCell):
    def initial_state(self, case_shape, /, *, dtype):
        return jnp.zeros(case_shape + (1,), dtype=dtype)

    def step(self, state, inputs, /, *, key=None):
        del key
        next_state = state + inputs
        return next_state, next_state

    def step_with_context(
        self,
        state,
        inputs,
        /,
        *,
        time,
        interval,
        key=None,
    ):
        del key
        next_state = state + inputs * (time + interval)[..., None]
        return next_state, next_state


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


def test_bidirectional_time_aware_recurrence_matches_explicit_reverse_time():
    cell = _PhysicalVectorAccumulator()
    model = BidirectionalRecurrentSequenceModel(cell, cell)
    inputs = jnp.ones((4, 1))
    valid = jnp.ones((4,), dtype=bool)
    times = jnp.asarray((0.0, 1.0, 4.0, 10.0))
    batch = RecurrentBatch(inputs, valid, time=times)

    output = model(batch)
    backward_reference = run_recurrent(
        cell,
        RecurrentBatch(
            inputs[::-1],
            valid,
            time=times[::-1],
            time_direction="backward",
        ),
    ).outputs[::-1]

    assert jnp.array_equal(output[..., :1], run_recurrent(cell, batch).outputs)
    assert jnp.array_equal(output[..., 1:], backward_reference)
    assert jnp.array_equal(
        backward_reference[..., 0],
        jnp.asarray((25.0, 24.0, 20.0, 10.0)),
    )


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


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    (
        ({"backbone_depth": -1}, ValueError, "nonnegative integer"),
        ({"backbone_depth": 1.5}, ValueError, "nonnegative integer"),
        (
            {"backbone_depth": 0, "backbone_width": 4},
            ValueError,
            "must be None",
        ),
        ({"backbone_width": 0}, ValueError, "positive integer"),
        ({"activation": None}, TypeError, "callable"),
        ({"dtype": jnp.complex64}, TypeError, "real floating"),
    ),
)
def test_cfc_cell_rejects_invalid_construction(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        CfCCell(2, 3, **kwargs, key=jr.key(13))


def test_cfc_cell_matches_full_gated_event_equation():
    cell = CfCCell(2, 2, backbone_depth=0, dtype=jnp.float64, key=jr.key(13))
    candidate_weight = jnp.asarray(
        (
            ((0.2, -0.1, 0.3, 0.4), (-0.3, 0.2, 0.1, -0.2)),
            ((-0.1, 0.5, -0.2, 0.3), (0.4, -0.2, 0.2, 0.1)),
        ),
        dtype=jnp.float64,
    )
    candidate_bias = jnp.asarray(((0.1, -0.2), (0.05, 0.3)))
    time_weight = jnp.asarray(
        (
            ((0.3, 0.2, -0.1, 0.4), (-0.2, 0.1, 0.5, -0.3)),
            ((-0.4, 0.2, 0.3, 0.1), (0.1, -0.5, 0.2, 0.4)),
        ),
        dtype=jnp.float64,
    )
    time_bias = jnp.asarray(((0.2, -0.1), (-0.3, 0.25)))
    cell = eqx.tree_at(
        lambda current: (
            current.candidate_weight,
            current.candidate_bias,
            current.time_weight,
            current.time_bias,
        ),
        cell,
        (candidate_weight, candidate_bias, time_weight, time_bias),
    )
    inputs = jnp.asarray(((0.2, -0.4), (0.1, 0.3)))
    state = jnp.asarray(((0.5, -0.1), (-0.2, 0.4)))
    interval = jnp.asarray((0.0, 1.5))

    next_state, output = cell.step_with_context(
        state,
        inputs,
        time=jnp.asarray((2.0, 9.0)),
        interval=interval,
    )
    features = jnp.concatenate((inputs, state), axis=-1)
    candidates = jnp.tanh(
        jnp.einsum("koi,...i->...ko", candidate_weight, features) + candidate_bias
    )
    time_parameters = jnp.einsum("koi,...i->...ko", time_weight, features) + time_bias
    gate = jax.nn.sigmoid(
        time_parameters[..., 0, :] * interval[..., None] + time_parameters[..., 1, :]
    )
    expected = candidates[..., 0, :] * (1.0 - gate) + candidates[..., 1, :] * gate

    assert jnp.allclose(next_state, expected)
    assert jnp.array_equal(output, next_state)
    assert not jnp.allclose(next_state[0], state[0])
    assert jnp.all(jnp.abs(next_state) <= 1.0)

    regular, _ = cell.step(state, inputs)
    physical_unit, _ = cell.step_with_context(
        state,
        inputs,
        time=jnp.asarray((20.0, -7.0)),
        interval=jnp.ones((2,)),
    )
    assert jnp.allclose(regular, physical_unit)


def test_cfc_sequence_is_jittable_differentiable_and_respects_packing():
    cell = CfCCell(
        2,
        3,
        backbone_width=4,
        backbone_depth=2,
        dtype=jnp.float64,
        key=jr.key(14),
    )
    inputs = jr.normal(jr.key(15), (2, 5, 2), dtype=jnp.float64)
    valid = jnp.asarray(
        ((True, True, True, False, False), (True, True, True, True, True))
    )
    reset = jnp.zeros_like(valid).at[1, 3].set(True)
    time = jnp.asarray(
        ((0.0, 0.4, 1.1, 1.1, 1.1), (0.0, 0.2, 0.2, 0.9, 2.0)),
        dtype=jnp.float64,
    )
    batch = RecurrentBatch(inputs, valid, reset=reset, time=time)

    result = eqx.filter_jit(lambda current: run_recurrent(cell, current))(batch)
    assert result.outputs.shape == (2, 5, 3)
    assert result.states.shape == (2, 5, 3)
    assert jnp.array_equal(result.outputs[0, 3:], jnp.zeros((2, 3)))
    assert jnp.array_equal(result.states[0, 3], result.states[0, 2])
    assert jnp.array_equal(result.states[0, 4], result.states[0, 2])
    assert jnp.all(jnp.abs(result.states) <= 1.0)

    gradient = jax.grad(
        lambda values: jnp.sum(
            run_recurrent(
                cell,
                RecurrentBatch(values, valid, reset=reset, time=time),
            ).outputs
            ** 2
        )
    )(inputs)
    assert jnp.all(jnp.isfinite(gradient))
    parameter_gradient = eqx.filter_grad(
        lambda current: jnp.sum(run_recurrent(current, batch).outputs ** 2)
    )(cell)
    assert jax.tree.leaves(parameter_gradient)
    assert all(
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(parameter_gradient)
    )


def test_cfc_streaming_preserves_the_boundary_interval():
    cell = CfCCell(2, 3, dtype=jnp.float64, key=jr.key(16))
    inputs = jr.normal(jr.key(17), (6, 2), dtype=jnp.float64)
    valid = jnp.ones((6,), dtype=bool)
    time = jnp.asarray((0.0, 0.2, 0.8, 1.7, 3.1, 5.0))
    split = 3

    whole = run_recurrent(cell, RecurrentBatch(inputs, valid, time=time))
    first = run_recurrent(
        cell,
        RecurrentBatch(inputs[:split], valid[:split], time=time[:split]),
    )
    second = run_recurrent(
        cell,
        RecurrentBatch(inputs[split:], valid[split:], time=time[split:]),
        initial_state=first.final_state,
        initial_context=first.final_context,
    )

    assert jnp.allclose(
        jnp.concatenate((first.outputs, second.outputs)),
        whole.outputs,
    )
    assert jnp.allclose(
        jnp.concatenate((first.states, second.states)),
        whole.states,
    )


def test_stacked_recurrent_cell_forwards_context_to_nested_cfc_cells():
    first_cell = CfCCell(
        1,
        2,
        backbone_depth=0,
        dtype=jnp.float64,
        key=jr.key(18),
    )
    second_cell = CfCCell(
        2,
        1,
        backbone_depth=0,
        use_bias=False,
        dtype=jnp.float64,
        key=jr.key(19),
    )
    stack = StackedRecurrentCell((first_cell, second_cell))
    inputs = jnp.asarray(((0.1,), (0.2,), (-0.1,), (0.4,)))
    valid = jnp.ones((4,), dtype=bool)
    time = jnp.asarray((0.0, 0.3, 1.1, 2.8))
    batch = RecurrentBatch(inputs, valid, time=time)

    first = run_recurrent(first_cell, batch)
    second = run_recurrent(
        second_cell,
        RecurrentBatch(first.outputs, valid, time=time),
    )
    stacked = run_recurrent(stack, batch)

    assert jnp.allclose(stacked.states[0], first.states)
    assert jnp.allclose(stacked.states[1], second.states)
    assert jnp.allclose(stacked.outputs, second.outputs)
