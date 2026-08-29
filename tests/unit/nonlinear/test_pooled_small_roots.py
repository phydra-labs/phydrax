import jax
import jax.numpy as jnp

import phydrax as phx


def test_pooled_small_roots_preserve_task_order_and_lane_invariance():
    targets = jnp.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
    initial = jnp.ones_like(targets)

    def residual(value, target):
        return value**2 - target

    full = phx.nonlinear.batched_small_root(
        residual,
        initial,
        targets,
        maximum_steps=16,
    )
    pooled = phx.nonlinear.pooled_small_root(
        residual,
        initial,
        targets,
        lane_count=2,
        maximum_steps=16,
    )
    assert jnp.allclose(pooled.result.state, full.state, atol=1e-6)
    assert jnp.array_equal(pooled.result.status, full.status)
    assert sorted(pooled.evidence.completion_order.tolist()) == list(range(5))
    assert int(pooled.evidence.refills) == 3
    assert 0.0 < float(pooled.evidence.utilization) <= 1.0


def test_pooled_small_roots_are_jittable_and_record_nonfinite_failures():
    targets = jnp.asarray([[1.0], [jnp.nan], [9.0]])
    initial = jnp.ones_like(targets)

    kernel = phx.nonlinear.SmallRootKernel(
        lambda value, target: value**2 - target,
        maximum_steps=8,
    )
    solve = jax.jit(
        lambda values, args: kernel.solve_pooled(
            values,
            args,
            lane_count=2,
        )
    )
    result = solve(initial, targets)
    assert result.result.status[1] == int(
        phx.nonlinear.NonlinearStatus.NONFINITE_EVALUATION
    )
    assert sorted(result.evidence.completion_order.tolist()) == [0, 1, 2]
