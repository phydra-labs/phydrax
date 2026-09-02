import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.layers import (
    AbstractRecurrentCell,
    AbstractTimeAwareAssociativeRecurrence,
    AbstractTimeAwareRecurrentCell,
    AffineRecurrence,
    RecurrentBatch,
    run_affine_recurrence,
    run_associative_recurrence,
    run_recurrent,
)
from phydrax.nn.models import CausalCoordinateNetwork, CausalCoordinatePlan


class _RandomAccumulator(AbstractRecurrentCell):
    width: int = eqx.field(static=True)

    def __init__(self, width):
        self.width = int(width)

    def initial_state(self, case_shape, /, *, dtype):
        return jnp.zeros(case_shape + (self.width,), dtype=dtype)

    def step(self, state, inputs, /, *, key=None):
        noise = jnp.zeros_like(inputs) if key is None else jr.normal(key, inputs.shape)
        next_state = state + inputs + noise
        return next_state, 2.0 * next_state


class _PhysicalAccumulator(AbstractTimeAwareRecurrentCell):
    def initial_state(self, case_shape, /, *, dtype):
        return jnp.zeros(case_shape, dtype=dtype)

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
        next_state = state + inputs * (time + interval)
        return next_state, next_state


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


class _PhysicalAdditiveRecurrence(AbstractTimeAwareAssociativeRecurrence):
    def initial_state(self, case_shape, /, *, dtype):
        return jnp.zeros(case_shape, dtype=dtype)

    def identity(self, case_shape, /, *, dtype):
        return jnp.zeros(case_shape, dtype=dtype)

    def encode_step(self, inputs, /, *, key=None):
        del key
        return inputs

    def encode_step_with_context(
        self,
        inputs,
        /,
        *,
        time,
        interval,
        key=None,
    ):
        del key
        return inputs * (time + interval)

    def combine(self, left, right, /):
        return left + right

    def apply_prefix(self, state, summary, /):
        return state + summary


def _packed_affine_batch():
    transitions = jnp.asarray(
        [
            [[0.8, 0.7], [0.6, 0.5], [0.4, 0.3], [9.0, 9.0], [9.0, 9.0]],
            [[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2], [0.1, 0.05]],
        ]
    )
    additions = jnp.arange(20, dtype=float).reshape(2, 5, 2) / 10.0
    valid = jnp.asarray(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    reset = jnp.asarray(
        [[False, False, False, False, False], [False, False, True, False, False]]
    )
    return RecurrentBatch((transitions, additions), valid, reset=reset)


def test_recurrent_batch_rejects_ambiguous_padding_and_resets():
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="after padding"):
        batch = RecurrentBatch(
            jnp.ones((4, 2)),
            jnp.asarray([True, False, True, False]),
        )
        jax.block_until_ready(batch.valid)

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="requires a valid"):
        batch = RecurrentBatch(
            jnp.ones((3, 2)),
            jnp.asarray([True, False, False]),
            reset=jnp.asarray([False, True, False]),
        )
        jax.block_until_ready(batch.valid)


def test_affine_serial_and_associative_execution_match_with_resets_and_padding():
    recurrence = AffineRecurrence(jnp.asarray([1.0, -0.5]))
    batch = _packed_affine_batch()

    serial = jax.jit(
        lambda current: run_affine_recurrence(recurrence, current, execution="serial")
    )(batch)
    associative = jax.jit(
        lambda current: run_affine_recurrence(
            recurrence, current, execution="associative"
        )
    )(batch)

    assert jnp.allclose(serial.states, associative.states)
    assert jnp.allclose(serial.outputs, associative.outputs)
    assert jnp.allclose(serial.final_state, associative.final_state)
    assert jnp.array_equal(serial.outputs[0, 3:], jnp.zeros((2, 2)))
    assert jnp.array_equal(serial.states[0, 3], serial.states[0, 2])
    assert jnp.array_equal(serial.states[0, 4], serial.states[0, 2])

    transition, addition = batch.inputs
    reset_step = transition[1, 2] * recurrence.initial + addition[1, 2]
    assert jnp.allclose(serial.states[1, 2], reset_step)


def test_affine_chunking_preserves_canonical_reset_state():
    recurrence = AffineRecurrence(jnp.asarray([1.0, -0.5]))
    transitions = jnp.asarray(
        [
            [0.8, 0.7],
            [0.6, 0.5],
            [0.4, 0.3],
            [0.9, 0.8],
            [0.7, 0.6],
            [0.5, 0.4],
        ]
    )
    additions = jnp.arange(12, dtype=float).reshape(6, 2) / 10.0
    valid = jnp.ones((6,), dtype=bool)
    reset = jnp.asarray([False, False, False, True, False, False])
    initial_state = jnp.asarray([4.0, -3.0])

    for execution in ("serial", "associative"):
        whole = run_affine_recurrence(
            recurrence,
            RecurrentBatch((transitions, additions), valid, reset=reset),
            initial_state=initial_state,
            execution=execution,
        )
        first = run_affine_recurrence(
            recurrence,
            RecurrentBatch((transitions[:2], additions[:2]), valid[:2]),
            initial_state=initial_state,
            execution=execution,
        )
        second = run_affine_recurrence(
            recurrence,
            RecurrentBatch(
                (transitions[2:], additions[2:]),
                valid[2:],
                reset=reset[2:],
            ),
            initial_state=first.final_state,
            execution=execution,
        )
        assert jnp.allclose(
            jnp.concatenate((first.states, second.states)),
            whole.states,
        )


def test_affine_composition_order_matches_dense_matrix_serial_execution():
    recurrence = AffineRecurrence(jnp.asarray([0.3, -0.2]), mode="matrix")
    transitions = jnp.asarray(
        [
            [[1.0, 2.0], [0.0, 1.0]],
            [[0.5, 0.0], [-1.0, 2.0]],
            [[1.0, -0.25], [0.5, 1.0]],
        ]
    )
    additions = jnp.asarray([[0.1, 0.2], [-0.3, 0.4], [0.2, -0.1]])
    batch = RecurrentBatch((transitions, additions), jnp.ones(3, dtype=bool))

    serial = run_affine_recurrence(recurrence, batch, execution="serial")
    associative = run_affine_recurrence(recurrence, batch, execution="associative")
    assert jnp.allclose(serial.states, associative.states, atol=1e-6, rtol=1e-6)


def test_affine_execution_has_matching_finite_gradients_and_vmap_behavior():
    recurrence = AffineRecurrence(jnp.asarray([0.2, -0.1]))
    transitions = jnp.asarray([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])
    valid = jnp.asarray([True, True, True])

    def loss(additions, execution):
        batch = RecurrentBatch((transitions, additions), valid)
        result = run_affine_recurrence(recurrence, batch, execution=execution)
        return jnp.sum(result.outputs**2)

    additions = jnp.ones((3, 2)) * 0.1
    serial_gradient = jax.jit(jax.grad(lambda value: loss(value, "serial")))(additions)
    associative_gradient = jax.jit(jax.grad(lambda value: loss(value, "associative")))(
        additions
    )
    assert jnp.all(jnp.isfinite(serial_gradient))
    assert jnp.allclose(serial_gradient, associative_gradient, atol=1e-5, rtol=1e-5)

    batched = jax.vmap(
        lambda value: (
            run_affine_recurrence(
                recurrence,
                RecurrentBatch((transitions, value), valid),
                execution="serial",
            ).outputs
        )
    )(jnp.stack((additions, 2.0 * additions)))
    assert batched.shape == (2, 3, 2)


def test_generic_recurrent_cell_masks_outputs_and_propagates_keys_deterministically():
    cell = _RandomAccumulator(3)
    inputs = jnp.ones((4, 3))
    valid = jnp.asarray([True, True, False, False])
    batch = RecurrentBatch(inputs, valid)
    key = jr.key(12)

    result = jax.jit(lambda current: run_recurrent(cell, current, key=key))(batch)
    repeated = run_recurrent(cell, batch, key=key)

    assert jnp.array_equal(result.outputs, repeated.outputs)
    assert jnp.array_equal(result.outputs[2:], jnp.zeros((2, 3)))
    assert jnp.array_equal(result.states[2], result.states[1])
    assert jnp.array_equal(result.final_state, result.states[-1])


def test_physical_time_and_intervals_reach_serial_and_associative_dispatch():
    inputs = jnp.ones((3,))
    valid = jnp.ones((3,), dtype=bool)

    def evaluate(times):
        batch = RecurrentBatch(inputs, valid, time=times)
        serial = run_recurrent(_PhysicalAccumulator(), batch)
        associative = run_associative_recurrence(
            _PhysicalAdditiveRecurrence(),
            batch,
        )
        return serial.outputs, associative.outputs

    serial, associative = jax.jit(evaluate)(jnp.asarray((1.0, 2.0, 5.0)))
    changed_serial, changed_associative = jax.jit(evaluate)(jnp.asarray((1.0, 4.0, 5.0)))
    network = CausalCoordinateNetwork(_PhysicalVectorAccumulator())
    causal = network(
        CausalCoordinatePlan(jnp.asarray((1.0, 2.0, 5.0))),
        jnp.ones((3, 1)),
    )
    changed_causal = network(
        CausalCoordinatePlan(jnp.asarray((1.0, 4.0, 5.0))),
        jnp.ones((3, 1)),
    )

    assert jnp.allclose(serial, jnp.asarray((1.0, 4.0, 12.0)))
    assert jnp.allclose(associative, serial)
    assert not jnp.allclose(changed_serial, serial)
    assert jnp.allclose(changed_associative, changed_serial)
    assert jnp.allclose(causal.ordered_outputs[..., 0], serial)
    assert jnp.allclose(changed_causal.ordered_outputs[..., 0], changed_serial)


def test_time_aware_streaming_context_preserves_the_boundary_interval_exactly():
    inputs = jnp.ones((5,))
    valid = jnp.ones((5,), dtype=bool)
    times = jnp.asarray((0.0, 1.0, 3.0, 6.0, 10.0))
    split = 2

    def evaluate(executor, recurrence):
        whole = executor(
            recurrence,
            RecurrentBatch(inputs, valid, time=times),
        )
        first = executor(
            recurrence,
            RecurrentBatch(inputs[:split], valid[:split], time=times[:split]),
        )
        second = executor(
            recurrence,
            RecurrentBatch(inputs[split:], valid[split:], time=times[split:]),
            initial_state=first.final_state,
            initial_context=first.final_context,
        )
        return whole, first, second

    serial, first_serial, second_serial = evaluate(
        run_recurrent,
        _PhysicalAccumulator(),
    )
    associative, first_associative, second_associative = evaluate(
        run_associative_recurrence,
        _PhysicalAdditiveRecurrence(),
    )

    assert jnp.array_equal(
        jnp.concatenate((first_serial.outputs, second_serial.outputs)),
        serial.outputs,
    )
    assert jnp.array_equal(
        jnp.concatenate((first_associative.outputs, second_associative.outputs)),
        associative.outputs,
    )
    assert jnp.array_equal(second_serial.outputs, second_associative.outputs)
    assert jnp.array_equal(second_serial.final_context.time, times[-1])
    assert bool(second_serial.final_context.has_time)


def test_causal_coordinate_streaming_preserves_physical_boundary_context():
    network = CausalCoordinateNetwork(_PhysicalVectorAccumulator())
    times = jnp.asarray((0.0, 1.0, 3.0, 6.0, 10.0))
    inputs = jnp.ones((5, 1))
    split = 2

    whole = network(CausalCoordinatePlan(times), inputs)
    first = network(CausalCoordinatePlan(times[:split]), inputs[:split])
    second = network(
        CausalCoordinatePlan(times[split:]),
        inputs[split:],
        initial_carry=first.final_carry,
        initial_context=first.final_context,
    )

    assert jnp.array_equal(
        jnp.concatenate((first.ordered_outputs, second.ordered_outputs)),
        whole.ordered_outputs,
    )
    assert second.final_context is second.recurrent.final_context
    assert jnp.array_equal(second.final_context.time, times[-1])
