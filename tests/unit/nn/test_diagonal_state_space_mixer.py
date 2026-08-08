#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _scalar_oracle_mixer(input_integration, *, rate=0.7):
    model = phx.nn.DiagonalStateSpaceMixer(
        state_size=1,
        input_integration=input_integration,
        initial_decay=rate,
        min_decay=1e-6,
        frequency_scale=0.0,
        key=jr.key(1),
    )
    raw_decay = jnp.log(jnp.expm1(rate - model.min_decay)).reshape((1,))
    return eqx.tree_at(
        lambda current: (
            current.raw_decay,
            current.frequencies,
            current.input_matrix_real,
            current.input_matrix_imag,
            current.output_matrix_real,
            current.output_matrix_imag,
            current.skip_matrix,
        ),
        model,
        (
            raw_decay,
            jnp.zeros((1,)),
            jnp.asarray([[0.5]]),
            jnp.zeros((1, 1)),
            jnp.ones((1, 1)),
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1)),
        ),
    )


@pytest.mark.parametrize("input_integration", ("zoh", "linear"))
def test_recurrent_matches_direct_variable_step_convolution(input_integration):
    model = phx.nn.DiagonalStateSpaceMixer(
        in_channels=2,
        out_channels=3,
        state_size=5,
        input_integration=input_integration,
        key=jr.key(2),
    )
    values = jr.normal(jr.key(3), (2, 9, 2))
    times = jnp.array(
        [
            [0.0, 0.03, 0.2, 0.21, 0.9, 1.4, 1.41, 2.0, 3.7],
            [0.0, 0.4, 0.41, 0.8, 1.0, 1.7, 2.5, 2.51, 2.8],
        ]
    )
    initial_state = jr.normal(jr.key(4), (2, 5)) + 1j * jr.normal(jr.key(5), (2, 5))

    recurrent = model.recurrent(values, times, initial_state=initial_state)
    direct = model.direct_convolution(values, times, initial_state=initial_state)

    assert recurrent.shape == (2, 9, 3)
    assert jnp.allclose(recurrent, direct, rtol=2e-5, atol=2e-6)


def test_irregular_zoh_matches_continuous_time_analytic_oracle():
    rate = 0.7
    model = _scalar_oracle_mixer("zoh", rate=rate)
    times = jnp.array([0.0, 0.02, 0.31, 0.9, 0.91, 2.4])
    values = jnp.ones_like(times)

    result = model(values, times)
    expected = (1.0 - jnp.exp(-rate * times)) / rate

    assert jnp.allclose(result, expected, rtol=2e-6, atol=2e-7)
    assert model.discretization == "exact"
    assert model.approximation == "none"
    assert model.method_id == "diagonal-state-space-mixer/exact/zoh/recurrent"
    assert result.dtype == model.raw_decay.dtype


def test_zoh_and_linear_input_semantics_are_distinct_and_linear_is_exact_for_ramp():
    rate = 0.4
    times = jnp.array([0.0, 0.13, 0.6, 1.7])
    values = times
    zoh = _scalar_oracle_mixer("zoh", rate=rate)(values, times)
    linear = _scalar_oracle_mixer("linear", rate=rate)(values, times)
    expected = times / rate - (1.0 - jnp.exp(-rate * times)) / rate**2

    assert jnp.allclose(linear, expected, rtol=3e-6, atol=3e-7)
    assert not jnp.allclose(zoh[1:], linear[1:], rtol=1e-3, atol=1e-4)
    assert zoh[1] == 0.0
    assert linear[1] > 0.0


def test_ragged_prefix_masks_ignore_padded_times_and_inputs():
    model = phx.nn.DiagonalStateSpaceMixer(
        in_channels=2,
        out_channels=2,
        state_size=4,
        input_integration="linear",
        key=jr.key(6),
    )
    values = jr.normal(jr.key(7), (2, 6, 2))
    times = jnp.array(
        [
            [0.0, 0.2, 0.7, jnp.nan, jnp.nan, jnp.nan],
            [0.0, 0.1, 0.4, 0.8, 1.3, 2.0],
        ]
    )
    mask = jnp.array(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, True, True],
        ]
    )

    result = model((values, times, mask))
    first_prefix = model(values[0, :3], times[0, :3])

    assert jnp.allclose(result[0, :3], first_prefix, rtol=2e-6, atol=2e-6)
    assert jnp.array_equal(result[0, 3:], jnp.zeros((3, 2)))
    assert jnp.all(jnp.isfinite(result))


def test_strictly_stable_conjugate_poles_preserve_long_memory():
    model = _scalar_oracle_mixer("zoh", rate=1e-5)
    model = eqx.tree_at(
        lambda current: current.frequencies,
        model,
        jnp.asarray([0.017]),
    )
    times = jnp.linspace(0.0, 10_000.0, 4096)
    values = jnp.zeros_like(times)
    initial_state = jnp.ones((1,), dtype=jnp.complex64)

    result = model.recurrent(values, times, initial_state=initial_state)
    poles = model.continuous_poles()

    assert jnp.all(jnp.real(poles) < 0.0)
    assert jnp.allclose(poles[1:], jnp.conj(poles[:1]))
    assert jnp.all(jnp.isfinite(result))
    assert jnp.max(jnp.abs(result)) <= 2.0 + 1e-5
    assert jnp.abs(result[-1]) < jnp.abs(result[0])
    assert jnp.abs(result[-1]) > 0.5 * jnp.abs(result[0])


def test_compile_and_parameter_and_input_gradients_are_finite():
    model = phx.nn.DiagonalStateSpaceMixer(
        in_channels=2,
        out_channels=2,
        state_size=4,
        input_integration="linear",
        key=jr.key(8),
    )
    values = jr.normal(jr.key(9), (3, 8, 2))
    times = jnp.array([0.0, 0.04, 0.2, 0.5, 0.51, 0.9, 1.8, 2.0])
    compiled = eqx.filter_jit(lambda current, signal: current((signal, times)))

    eager = model((values, times))
    lowered = compiled(model, values)
    parameter_gradient = eqx.filter_grad(
        lambda current: jnp.sum(current((values, times)) ** 2)
    )(model)
    input_gradient = jax.grad(lambda signal: jnp.sum(model((signal, times)) ** 2))(values)
    gradient_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]

    assert jnp.allclose(lowered, eager, rtol=2e-6, atol=2e-6)
    assert gradient_leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in gradient_leaves)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.linalg.norm(parameter_gradient.raw_decay) > 0.0
    assert jnp.linalg.norm(input_gradient) > 0.0


@pytest.mark.parametrize("input_integration", ("zoh", "linear"))
def test_associative_execution_matches_recurrent_with_ragged_cases(input_integration):
    model = phx.nn.DiagonalStateSpaceMixer(
        in_channels=2,
        out_channels=2,
        state_size=6,
        input_integration=input_integration,
        execution="associative",
        key=jr.key(10),
    )
    values = jr.normal(jr.key(11), (2, 10, 2))
    times = jnp.array(
        [
            [0.0, 0.1, 0.11, 0.4, 0.9, 1.0, 1.7, 2.2, 3.0, 4.0],
            [0.0, 0.3, 0.5, 0.51, 0.7, 1.4, 2.0, 2.1, 2.7, 3.3],
        ]
    )
    mask = jnp.array(
        [
            [True, True, True, True, True, True, False, False, False, False],
            [True, True, True, True, True, True, True, True, True, True],
        ]
    )

    recurrent = model.recurrent(values, times, mask=mask)
    associative = model(values, times, mask=mask)

    assert jnp.allclose(associative, recurrent, rtol=3e-5, atol=3e-6)
    assert jnp.array_equal(associative[0, 6:], jnp.zeros((4, 2)))


def test_operator_batch_preserves_case_times_masks_and_prediction_contract():
    model = phx.nn.DiagonalStateSpaceMixer(
        in_channels=2,
        out_channels=3,
        state_size=4,
        source_key="signal",
        input_integration="linear",
        key=jr.key(12),
    )
    values = jr.normal(jr.key(13), (2, 5, 2))
    coordinates = jnp.array(
        [
            [[0.0], [0.1], [0.5], [1.2], [2.0]],
            [[0.0], [0.3], [0.31], [0.9], [1.7]],
        ]
    )
    mask = jnp.array([[True, True, True, False, False], [True, True, True, True, True]])
    batch = phx.nn.OperatorBatch(
        inputs={
            "signal": phx.nn.FunctionSamples(
                values=values, coordinates=coordinates, mask=mask
            )
        },
        queries={
            "query": phx.nn.FunctionSamples(
                values=None, coordinates=coordinates, mask=mask
            )
        },
        case_axes=("case",),
    )

    result = model(batch)
    direct = model.recurrent(values, coordinates[..., 0], mask=mask)
    prediction = model.predict(batch)
    contract = model.operator_contract

    assert result.shape == (2, 5, 3)
    assert jnp.allclose(result, direct, rtol=2e-6, atol=2e-6)
    assert jnp.array_equal(result[0, 3:], jnp.zeros((2, 3)))
    assert jnp.allclose(prediction.field("output").values, result)
    assert contract.architecture == "DiagonalStateSpaceMixer"
    assert contract.capabilities.source_query_relations == ("coincident",)
    assert contract.capabilities.masks == "supported"
    configuration = dict(contract.configuration)
    assert configuration["input_integration"] == "linear"
    assert configuration["discretization"] == "exact"
    assert configuration["approximation"] == "none"
    assert configuration["method_id"] == model.method_id


def test_dense_reference_rejects_lengths_above_configured_bound():
    model = phx.nn.DiagonalStateSpaceMixer(
        state_size=2,
        max_direct_length=3,
        key=jr.key(14),
    )
    with pytest.raises(ValueError, match="max_direct_length"):
        model.direct_convolution(jnp.ones((4,)), jnp.arange(4.0))


def test_stiff_decay_initialization_and_repeated_nodes_remain_finite():
    model = phx.nn.DiagonalStateSpaceMixer(
        state_size=2,
        initial_decay=1_000.0,
        min_decay=1e-4,
        frequency_scale=0.0,
        key=jr.key(15),
    )

    result = model(jnp.ones((2,)), jnp.zeros((2,)))

    assert jnp.all(jnp.isfinite(model.raw_decay))
    assert jnp.all(jnp.isfinite(model.continuous_poles()))
    assert jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize(
    "parameters",
    (
        {"initial_decay": jnp.nan},
        {"min_decay": jnp.nan},
        {"frequency_scale": jnp.nan},
        {"initial_decay": 2e-50, "min_decay": 1e-50, "dtype": jnp.float32},
    ),
)
def test_nonfinite_or_unrepresentable_stability_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        phx.nn.DiagonalStateSpaceMixer(**parameters)


def test_linear_input_coefficient_does_not_overflow_for_large_finite_decay():
    model = _scalar_oracle_mixer("linear")
    model = eqx.tree_at(
        lambda current: current.raw_decay,
        model,
        jnp.asarray([1e20], dtype=model.raw_decay.dtype),
    )

    result = model(jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 1.0]))

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result[-1], jnp.asarray(1e-20), rtol=2e-5, atol=0.0)


def test_scalar_length_one_operator_batch_preserves_singleton_case_axis():
    model = phx.nn.DiagonalStateSpaceMixer(
        state_size=2,
        source_key="signal",
        key=jr.key(16),
    )
    coordinates = jnp.zeros((1, 1, 1))
    mask = jnp.ones((1, 1), dtype=bool)
    batch = phx.nn.OperatorBatch(
        inputs={
            "signal": phx.nn.FunctionSamples(
                values=jnp.ones((1, 1)),
                coordinates=coordinates,
                mask=mask,
            )
        },
        queries={
            "query": phx.nn.FunctionSamples(
                values=None,
                coordinates=coordinates,
                mask=mask,
            )
        },
        case_axes=("case",),
    )

    result = model(batch)

    assert result.shape == (1, 1)
