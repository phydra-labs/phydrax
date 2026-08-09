import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


@pytest.mark.parametrize("input_integration", ("zoh", "linear"))
def test_selective_mixer_serial_and_associative_execution_are_equivalent(
    input_integration,
):
    model = phx.nn.operator.architectures.SelectiveStateSpaceMixer(
        in_channels=3,
        out_channels=2,
        state_size=7,
        input_integration=input_integration,
        key=jr.key(0),
    )
    values = jr.normal(jr.key(1), (3, 17, 3))
    times = jnp.array(
        [
            0.0,
            0.01,
            0.08,
            0.2,
            0.21,
            0.5,
            0.9,
            0.91,
            1.4,
            2.0,
            2.01,
            2.8,
            3.0,
            4.2,
            4.21,
            5.0,
            7.0,
        ]
    )
    initial = jr.normal(jr.key(2), (3, 7))

    recurrent = model.recurrent(values, times, initial_state=initial)
    associative = model.associative(values, times, initial_state=initial)

    assert recurrent.shape == (3, 17, 2)
    assert jnp.allclose(associative, recurrent, rtol=2e-5, atol=2e-6)
    assert jnp.all(model.decay_rates() > 0.0)


def test_selective_mixer_packed_resets_and_diagnostics_are_explicit():
    model = phx.nn.operator.architectures.SelectiveStateSpaceMixer(
        in_channels=2,
        out_channels=2,
        state_size=5,
        input_integration="linear",
        execution="associative",
        training_delta_range=(0.1, 0.4),
        key=jr.key(3),
    )
    values = jr.normal(jr.key(4), (6, 2))
    times = jnp.array([0.0, 0.2, jnp.nan, 0.0, 0.7, jnp.nan])
    mask = jnp.array([True, True, False, True, True, False])
    reset = jnp.array([False, False, False, True, False, False])

    output, diagnostics = model.evaluate_with_diagnostics(
        values, times, mask=mask, reset=reset
    )
    first_segment = model.associative(values[:2], times[:2])
    second_segment = model.associative(values[3:5], times[3:5])

    assert jnp.allclose(output[:2], first_segment, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(output[3:5], second_segment, rtol=2e-5, atol=2e-6)
    streamed_segment = model.associative(
        values[3:5],
        times[3:5],
        reset=jnp.array([True, False]),
        initial_state=jnp.ones((model.state_size,)),
    )
    assert jnp.allclose(streamed_segment, second_segment, rtol=2e-5, atol=2e-6)
    assert jnp.array_equal(output[jnp.array([2, 5])], jnp.zeros((2, 2)))
    assert diagnostics.interval_count == 2
    assert diagnostics.segment_count == 2
    assert diagnostics.minimum_physical_step == pytest.approx(0.2)
    assert diagnostics.maximum_physical_step == pytest.approx(0.7)
    assert diagnostics.extrapolated_fraction == pytest.approx(0.5)
    assert diagnostics.minimum_effective_step > 0.0
    assert diagnostics.maximum_effective_step >= diagnostics.minimum_effective_step


def test_selective_mixer_requires_resets_after_padding():
    model = phx.nn.operator.architectures.SelectiveStateSpaceMixer(key=jr.key(5))
    values = jnp.ones((4,))
    times = jnp.array([0.0, jnp.nan, 0.0, 0.2])
    mask = jnp.array([True, False, True, True])

    with pytest.raises(Exception, match="must declare reset"):
        model(values, times, mask=mask)

    with pytest.raises(Exception, match="reset=True requires a valid physical sample"):
        model(
            values,
            times,
            mask=mask,
            reset=jnp.array([False, True, False, False]),
        )


def test_selective_mixer_jit_and_parameter_input_gradients_are_finite():
    model = phx.nn.operator.architectures.SelectiveStateSpaceMixer(
        in_channels=2,
        out_channels=2,
        state_size=6,
        input_integration="linear",
        key=jr.key(6),
    )
    values = jr.normal(jr.key(7), (2, 13, 2))
    times = jnp.array(
        [0.0, 0.03, 0.2, 0.21, 0.8, 1.1, 1.11, 1.9, 2.7, 2.71, 3.8, 5.0, 8.0]
    )
    compiled = eqx.filter_jit(lambda current, signal: current(signal, times))

    eager = model(values, times)
    lowered = compiled(model, values)
    parameter_gradient = eqx.filter_grad(
        lambda current: jnp.sum(current(values, times) ** 2)
    )(model)
    input_gradient = jax.grad(lambda signal: jnp.sum(model(signal, times) ** 2))(values)
    gradient_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]

    assert jnp.allclose(lowered, eager, rtol=2e-5, atol=2e-6)
    assert gradient_leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in gradient_leaves)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.linalg.norm(parameter_gradient.delta_weight) > 0.0
    assert jnp.linalg.norm(parameter_gradient.input_gate_weight) > 0.0
    assert jnp.linalg.norm(parameter_gradient.output_gate_weight) > 0.0


def test_selective_mixer_operator_batch_contract_preserves_masks():
    model = phx.nn.operator.architectures.SelectiveStateSpaceMixer(
        in_channels=2,
        out_channels=3,
        state_size=5,
        source_key="signal",
        input_integration="zoh",
        key=jr.key(8),
    )
    values = jr.normal(jr.key(9), (2, 5, 2))
    coordinates = jnp.array(
        [
            [[0.0], [0.1], [0.5], [1.2], [2.0]],
            [[0.0], [0.3], [0.31], [0.9], [1.7]],
        ]
    )
    mask = jnp.array([[True, True, True, False, False], [True, True, True, True, True]])
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "signal": phx.nn.operator.FunctionSamples(
                values=values, coordinates=coordinates, mask=mask
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None, coordinates=coordinates, mask=mask
            )
        },
        case_axes=("case",),
    )

    output = model(batch)
    direct = model(values, coordinates[..., 0], mask=mask)
    contract = model.operator_contract

    assert output.shape == (2, 5, 3)
    assert jnp.allclose(output, direct, rtol=2e-5, atol=2e-6)
    assert jnp.array_equal(output[0, 3:], jnp.zeros((2, 3)))
    assert contract.architecture == "SelectiveStateSpaceMixer"
    assert contract.capabilities.masks == "supported"
    assert dict(contract.configuration)["method_id"] == model.method_id
