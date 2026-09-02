import jax
import jax.numpy as jnp

from phydrax.solver._segmented_execution import (
    FixedCapacitySegmentPolicy,
    FixedCapacitySegmentStep,
    run_fixed_capacity_segments,
)


def test_fixed_capacity_segments_jit_runtime_count_and_neutral_tail():
    policy = FixedCapacitySegmentPolicy(4, 3, 0)

    def run(initial):
        def advance(carry, index):
            del index
            next_carry = carry + 1
            return FixedCapacitySegmentStep(
                next_carry,
                carry.astype(float),
                next_carry.astype(float),
                1,
                0,
                next_carry >= 2,
                0,
            )

        return run_fixed_capacity_segments(policy, initial, advance)

    carry, evidence = jax.jit(run)(jnp.asarray(0, dtype=jnp.int32))
    assert carry == 2
    assert evidence.segment_count == 2
    assert not evidence.capacity_exceeded
    assert jnp.array_equal(evidence.active, jnp.asarray([True, True, False, False]))
    assert jnp.array_equal(evidence.step_counts, jnp.asarray([1, 1, 0, 0]))


def test_fixed_capacity_segments_fail_closed_on_exact_cap_without_terminal():
    policy = FixedCapacitySegmentPolicy(2, 1, 0, failure=17)

    def advance(carry, index):
        del index
        return FixedCapacitySegmentStep(carry + 1, carry, carry + 1, 1)

    _, evidence = run_fixed_capacity_segments(policy, jnp.asarray(0.0), advance)
    assert evidence.capacity_exceeded
    assert evidence.terminal_status == 17


def test_fixed_capacity_segments_enforce_cumulative_event_capacity():
    policy = FixedCapacitySegmentPolicy(3, 1, 1, failure=23)

    def advance(carry, index):
        del index
        next_carry = carry + 1
        return FixedCapacitySegmentStep(
            next_carry,
            carry.astype(float),
            next_carry.astype(float),
            1,
            1,
            next_carry >= 2,
            0,
        )

    _, evidence = run_fixed_capacity_segments(
        policy,
        jnp.asarray(0, dtype=jnp.int32),
        advance,
    )

    assert evidence.capacity_exceeded
    assert evidence.terminal_status == 23
    assert evidence.segment_count == 2
