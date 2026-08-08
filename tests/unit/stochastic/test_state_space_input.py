import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _two_case_problem(input_signal, *, args=None):
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((2, 1)),
        jnp.asarray([[1.0]]),
        state_shape=(1,),
        prior_id="input-prior",
    )
    transition = phx.stochastic.LinearGaussianTransitionKernel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.1]]),
        state_shape=(1,),
        process_id="input-transition",
    )
    observation = phx.stochastic.LinearGaussianObservationModel(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        state_shape=(1,),
        observation_shape=(1,),
    )
    model = phx.stochastic.StateSpaceModel(
        prior, transition, observation, model_id="input-model"
    )
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([[0.5, 1.5], [1.0, 2.0]]),
        jnp.zeros((2, 2, 1)),
        case_axes=("case",),
        case_shape=(2,),
        observation_axes=("sensor",),
        case_ids=("first", "second"),
        sequence_id="input-observations",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=jnp.asarray([0.0, 0.0]),
        input_signal=input_signal,
        args=args,
        problem_id="input-problem",
    )


def test_sampled_input_hand_checked_interpolation_and_right_continuity():
    times = jnp.asarray([0.0, 1.0, 3.0])
    values = jnp.asarray([[0.0], [10.0], [30.0]])
    hold = phx.stochastic.SampledStateSpaceInput(
        times,
        values,
        interpolation="zero-order-hold",
        input_id="hold",
    )
    linear = phx.stochastic.SampledStateSpaceInput(
        times,
        values,
        interpolation="linear",
        input_id="linear",
    )
    queries = jnp.asarray([0.5, 1.0, 2.5, 3.0])

    hold_values = jax.jit(jax.vmap(lambda time: hold.evaluate(time, 0).value))(queries)
    linear_evaluations = jax.vmap(linear.evaluate, in_axes=(0, None))(queries, 0)

    assert isinstance(hold, phx.stochastic.AbstractStateSpaceInput)
    assert isinstance(hold.evaluate(0.5, 0), phx.stochastic.InputEvaluation)
    assert jnp.allclose(hold_values[:, 0], jnp.asarray([0.0, 10.0, 10.0, 30.0]))
    assert jnp.all(linear_evaluations.valid)
    assert jnp.allclose(
        linear_evaluations.value[:, 0], jnp.asarray([5.0, 10.0, 25.0, 30.0])
    )
    assert jnp.allclose(hold.evaluate(jnp.nextafter(1.0, 0.0), 0).value, 0.0)
    assert jnp.allclose(hold.evaluate(1.0, 0).value, 10.0)
    assert not hold.evaluate(-0.1, 0).valid


def test_sampled_input_handles_per_case_irregular_knots_and_padding_masks():
    signal = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray(
            [
                [0.0, 1.0, 3.0, 99.0],
                [-1.0, 0.5, 2.0, 123.0],
            ]
        ),
        jnp.asarray(
            [
                [[0.0], [10.0], [30.0], [jnp.nan]],
                [[0.0], [15.0], [30.0], [jnp.nan]],
            ]
        ),
        knot_valid=jnp.asarray([[True, True, True, False], [True, True, True, False]]),
        interpolation="linear",
        input_id="irregular",
    )

    evaluations = jax.jit(jax.vmap(signal.evaluate, in_axes=(0, 0)))(
        jnp.asarray([2.0, 1.25]), jnp.asarray([0, 1])
    )
    first_times, first_mask = signal.breakpoints(0.5, 2.5, 0)
    second_times, second_mask = signal.breakpoints(-0.5, 2.5, 1)

    assert signal.case_shape == (2,)
    assert jnp.all(evaluations.valid)
    assert jnp.allclose(evaluations.value[:, 0], jnp.asarray([20.0, 22.5]))
    assert jnp.array_equal(first_times, jnp.asarray([0.0, 1.0, 3.0, 99.0]))
    assert jnp.array_equal(first_mask, jnp.asarray([False, True, False, False]))
    assert jnp.array_equal(second_times, jnp.asarray([-1.0, 0.5, 2.0, 123.0]))
    assert jnp.array_equal(second_mask, jnp.asarray([False, True, True, False]))


def test_state_space_problem_rejects_input_without_schedule_support():
    unsupported = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray([[0.0, 2.0], [0.0, 1.5]]),
        jnp.asarray([[[0.0], [2.0]], [[0.0], [1.5]]]),
        interpolation="linear",
        input_id="short-support",
    )

    with pytest.raises(ValueError, match="support every active"):
        _two_case_problem(unsupported)


def test_bspline_input_values_and_coefficient_gradients_are_hand_checked():
    grid = phx.nn.BSplineGrid(jnp.asarray([0.0, 0.0, 1.0, 2.0, 2.0]), 1)
    signal = phx.stochastic.BSplineStateSpaceInput(
        grid,
        jnp.asarray([[0.0], [10.0], [20.0]]),
        input_id="piecewise-linear-spline",
    )
    queries = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
    evaluations = jax.jit(jax.vmap(signal.evaluate, in_axes=(0, None)))(queries, 0)

    def value_at_half(coefficients):
        candidate = eqx.tree_at(
            lambda current: current.coefficients, signal, coefficients
        )
        return candidate.evaluate(0.5, 0).value[0]

    gradient = jax.grad(value_at_half)(signal.coefficients)

    assert isinstance(signal, phx.stochastic.AbstractStateSpaceInput)
    assert jnp.all(evaluations.valid)
    assert jnp.allclose(
        evaluations.value[:, 0], jnp.asarray([0.0, 5.0, 10.0, 15.0, 20.0])
    )
    assert jnp.allclose(gradient[:, 0], jnp.asarray([0.5, 0.5, 0.0]))
    assert not signal.evaluate(2.1, 0).valid


def test_problem_step_context_exposes_schedule_inputs_and_arbitrary_evaluation():
    signal = phx.stochastic.SampledStateSpaceInput(
        jnp.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.5, 1.25, 2.0],
            ]
        ),
        jnp.asarray(
            [
                [[0.0], [10.0], [20.0], [30.0]],
                [[10.0], [15.0], [22.5], [30.0]],
            ]
        ),
        interpolation="linear",
        input_id="context-input",
    )
    problem = _two_case_problem(signal, args={"scale": jnp.asarray(3.0)})

    context = jax.jit(
        lambda case_index, step_index: problem.step_context(case_index, step_index)
    )(jnp.asarray(1), jnp.asarray(1))
    internal = context.evaluate_input(1.5)

    assert context.case_index == 1
    assert context.step_index == 1
    assert context.args["scale"] == 3.0
    assert context.input_valid
    assert jnp.allclose(context.transition_start_input, jnp.asarray([20.0]))
    assert jnp.allclose(context.transition_end_input, jnp.asarray([30.0]))
    assert jnp.allclose(context.observation_input, jnp.asarray([30.0]))
    assert isinstance(internal, phx.stochastic.InputEvaluation)
    assert internal.valid
    assert jnp.allclose(internal.value, jnp.asarray([25.0]))
    assert jnp.array_equal(context.input_breakpoints, jnp.asarray([0.0, 0.5, 1.25, 2.0]))
    assert jnp.array_equal(
        context.input_breakpoint_valid,
        jnp.asarray([False, False, True, False]),
    )


@pytest.mark.parametrize("input_kind", ("sampled", "bspline"))
@pytest.mark.parametrize("invalid_case_index", (-1, 2))
@pytest.mark.parametrize("use_jit", (False, True), ids=("eager", "jit"))
def test_state_space_input_case_indices_are_bounds_checked(
    input_kind, invalid_case_index, use_jit
):
    if input_kind == "sampled":
        signal = phx.stochastic.SampledStateSpaceInput(
            jnp.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
            jnp.asarray(
                [
                    [[0.0], [1.0], [2.0]],
                    [[10.0], [11.0], [12.0]],
                ]
            ),
            interpolation="linear",
            input_id="bounded-sampled",
        )
    else:
        signal = phx.stochastic.BSplineStateSpaceInput(
            phx.nn.BSplineGrid(jnp.asarray([0.0, 0.0, 1.0, 2.0, 2.0]), 1),
            jnp.asarray(
                [
                    [[0.0], [1.0], [2.0]],
                    [[10.0], [11.0], [12.0]],
                ]
            ),
            case_shape=(2,),
            input_id="bounded-bspline",
        )
    problem = _two_case_problem(signal)
    accessors = (
        lambda index: signal.evaluate(0.75, index).value,
        lambda index: signal.breakpoints(0.0, 1.5, index)[0],
        lambda index: problem.step_context(index, 0).observation_input,
    )

    for accessor in accessors:
        checked_accessor = eqx.filter_jit(accessor) if use_jit else accessor
        with pytest.raises(
            (ValueError, eqx.EquinoxRuntimeError),
            match="physical case index is out of bounds",
        ):
            value = checked_accessor(jnp.asarray(invalid_case_index))
            jax.block_until_ready(value)
