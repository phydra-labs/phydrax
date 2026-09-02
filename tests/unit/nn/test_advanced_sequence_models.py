import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.nn.layers import (
    LinearRecurrentUnit,
    RecurrentBatch,
    WeightSpaceRecurrence,
)
from phydrax.nn.models import LinearRecurrentModel, SelectiveSequenceModel


def _packed_sequence():
    values = jr.normal(jr.key(1), (2, 7, 3), dtype=jnp.float64)
    valid = jnp.array(
        [
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    reset = jnp.array(
        [
            [False, False, False, False, False, False, False],
            [False, False, False, True, False, False, False],
        ]
    )
    times = jnp.array(
        [
            [0.0, 0.1, 0.45, 0.9, 10.0, 11.0, 12.0],
            [0.0, 0.2, 0.7, 2.0, 2.0, 2.4, 3.1],
        ]
    )
    return RecurrentBatch(values, valid, reset=reset, time=times)


def test_linear_recurrent_unit_is_stable_and_serial_associative_exact():
    unit = LinearRecurrentUnit(
        3,
        5,
        output_size=2,
        dtype=jnp.float64,
        key=jr.key(2),
    )
    batch = _packed_sequence()
    serial = unit.evaluate_with_state(batch, execution="serial")
    associative = unit.evaluate_with_state(batch, execution="associative")

    assert jnp.max(jnp.abs(unit.eigenvalues())) < 1.0
    assert jnp.allclose(serial.states, associative.states, atol=2e-10, rtol=2e-10)
    assert jnp.allclose(serial.outputs, associative.outputs, atol=2e-10, rtol=2e-10)
    assert jnp.array_equal(serial.outputs[0, 4:], jnp.zeros((3, 2)))
    assert jnp.all(jnp.isfinite(serial.outputs))


def test_linear_recurrent_model_continues_exactly_across_sequence_chunks():
    unit = LinearRecurrentUnit(3, 4, dtype=jnp.float64, key=jr.key(3))
    model = LinearRecurrentModel(unit)
    values = jr.normal(jr.key(4), (8, 3), dtype=jnp.float64)
    whole_batch = RecurrentBatch(values, jnp.ones((8,), dtype=bool))
    whole = model.evaluate_with_state(whole_batch)

    first = model.evaluate_with_state(
        RecurrentBatch(values[:3], jnp.ones((3,), dtype=bool))
    )
    second = model.evaluate_with_state(
        RecurrentBatch(values[3:], jnp.ones((5,), dtype=bool)),
        initial_state=first.final_state,
    )
    assert jnp.allclose(
        jnp.concatenate((first.outputs, second.outputs)),
        whole.outputs,
        atol=2e-10,
        rtol=2e-10,
    )


def test_selective_sequence_has_serial_associative_parity_on_irregular_physical_time():
    batch = _packed_sequence()
    associative = SelectiveSequenceModel(
        3,
        4,
        inner_size=5,
        depth=2,
        execution="associative",
        dtype=jnp.float64,
        key=jr.key(5),
    )
    serial = SelectiveSequenceModel(
        3,
        4,
        inner_size=5,
        depth=2,
        execution="serial",
        dtype=jnp.float64,
        key=jr.key(5),
    )

    associative_output = associative(batch)
    serial_output = serial(batch)
    assert jnp.allclose(associative_output, serial_output, atol=2e-9, rtol=2e-9)
    assert jnp.array_equal(associative_output[0, 4:], jnp.zeros((3, 3)))
    assert jnp.all(jnp.isfinite(associative_output))


def test_selective_sequence_streaming_preserves_physical_time_and_convolution_state():
    model = SelectiveSequenceModel(
        3,
        4,
        inner_size=5,
        depth=1,
        dtype=jnp.float64,
        key=jr.key(6),
    )
    values = jr.normal(jr.key(7), (7, 3), dtype=jnp.float64)
    times = jnp.array([0.0, 0.1, 0.4, 0.9, 1.0, 1.7, 2.2])
    whole = model.evaluate_with_state(
        RecurrentBatch(values, jnp.ones((7,), dtype=bool), time=times)
    )
    first = model.evaluate_with_state(
        RecurrentBatch(values[:3], jnp.ones((3,), dtype=bool), time=times[:3])
    )
    second = model.evaluate_with_state(
        RecurrentBatch(values[3:], jnp.ones((4,), dtype=bool), time=times[3:]),
        initial_state=first.final_state,
    )

    assert jnp.allclose(
        jnp.concatenate((first.outputs, second.outputs)),
        whole.outputs,
        atol=2e-9,
        rtol=2e-9,
    )
    assert jnp.array_equal(second.final_state[0].last_time, times[-1])
    assert bool(second.final_state[0].has_time)


def test_selective_sequence_resets_isolate_segments_and_has_no_future_leakage():
    model = SelectiveSequenceModel(
        3,
        4,
        inner_size=4,
        depth=1,
        dtype=jnp.float64,
        key=jr.key(8),
    )
    values = jr.normal(jr.key(9), (8, 3), dtype=jnp.float64)
    times = jnp.array([0.0, 0.2, 0.7, 1.0, 3.0, 3.1, 3.6, 4.2])
    reset = jnp.array([False, False, False, False, True, False, False, False])
    packed = model(
        RecurrentBatch(values, jnp.ones((8,), dtype=bool), reset=reset, time=times)
    )
    first_chunk = model.evaluate_with_state(
        RecurrentBatch(
            values[:2],
            jnp.ones((2,), dtype=bool),
            reset=reset[:2],
            time=times[:2],
        )
    )
    second_chunk = model.evaluate_with_state(
        RecurrentBatch(
            values[2:],
            jnp.ones((6,), dtype=bool),
            reset=reset[2:],
            time=times[2:],
        ),
        initial_state=first_chunk.final_state,
    )
    assert jnp.allclose(
        jnp.concatenate((first_chunk.outputs, second_chunk.outputs)),
        packed,
        atol=2e-9,
        rtol=2e-9,
    )
    first = model(RecurrentBatch(values[:4], jnp.ones((4,), dtype=bool), time=times[:4]))
    second = model(RecurrentBatch(values[4:], jnp.ones((4,), dtype=bool), time=times[4:]))
    assert jnp.allclose(packed, jnp.concatenate((first, second)), atol=2e-9, rtol=2e-9)

    changed = values.at[5:].set(1000.0)
    changed_output = model(
        RecurrentBatch(changed, jnp.ones((8,), dtype=bool), reset=reset, time=times)
    )
    assert jnp.array_equal(changed_output[:5], packed[:5])


def test_recurrent_consumers_preserve_backward_time_direction_in_internal_batches():
    values = jr.normal(jr.key(20), (4, 3), dtype=jnp.float64)
    valid = jnp.ones((4,), dtype=bool)
    decreasing_times = jnp.asarray((6.0, 3.0, 1.0, 0.0))
    backward = RecurrentBatch(
        values,
        valid,
        time=decreasing_times,
        time_direction="backward",
    )
    directed_reference = RecurrentBatch(values, valid, time=-decreasing_times)

    unit = LinearRecurrentUnit(3, 4, dtype=jnp.float64, key=jr.key(21))
    assert jnp.array_equal(
        unit.evaluate_with_state(backward).outputs,
        unit.evaluate_with_state(directed_reference).outputs,
    )

    selective = SelectiveSequenceModel(
        3,
        3,
        inner_size=4,
        depth=2,
        dtype=jnp.float64,
        key=jr.key(22),
    )
    assert jnp.array_equal(selective(backward), selective(directed_reference))

    weight_space = WeightSpaceRecurrence(
        3,
        5,
        dtype=jnp.float64,
        key=jr.key(23),
    )
    center = jnp.zeros((5,), dtype=jnp.float64)
    assert jnp.array_equal(
        weight_space.evaluate_with_state(backward, center).outputs,
        weight_space.evaluate_with_state(directed_reference, center).outputs,
    )


def test_advanced_sequence_models_are_jittable_with_finite_input_gradients():
    batch = _packed_sequence()
    model = SelectiveSequenceModel(
        3,
        3,
        inner_size=4,
        depth=1,
        dtype=jnp.float64,
        key=jr.key(10),
    )
    compiled = eqx.filter_jit(lambda current: current(batch))(model)
    assert compiled.shape == (2, 7, 3)

    gradient = jax.grad(
        lambda values: jnp.sum(
            model(RecurrentBatch(values, batch.valid, reset=batch.reset, time=batch.time))
            ** 2
        )
    )(batch.inputs)
    assert jnp.all(jnp.isfinite(gradient))


def test_linear_recurrent_operator_adapts_coincident_masked_sequences():
    values = jr.normal(jr.key(11), (2, 7, 3), dtype=jnp.float64)
    coordinates = jnp.broadcast_to(
        jnp.linspace(0.0, 1.0, 7)[None, :, None],
        (2, 7, 1),
    )
    valid = jnp.array(
        [
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "history": phx.nn.operator.FunctionSamples(
                values=values,
                coordinates=coordinates,
                mask=valid,
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=coordinates,
                mask=valid,
            )
        },
        case_axes=("case",),
    )
    operator = phx.nn.operator.architectures.LinearRecurrentOperator(
        in_channels=3,
        out_channels=2,
        state_size=5,
        source_key="history",
        dtype=jnp.float64,
        key=jr.key(12),
    )
    prediction = eqx.filter_jit(lambda model: model(batch))(operator)
    assert prediction.shape == (2, 7, 2)
    assert jnp.array_equal(prediction[0, 4:], jnp.zeros((3, 2)))
    assert operator.operator_contract.architecture == "LinearRecurrentOperator"
    assert dict(operator.operator_contract.configuration)["time_semantics"] == (
        "ordered_samples"
    )
