import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_single_cosine_has_analytic_power_and_full_mode_count():
    count = 8
    x = (jnp.arange(count) + 0.5) / count
    field = jnp.cos(2.0 * jnp.pi * x)
    plan = phx.discretization.PeriodicFourierShellPlan(
        (count,),
        (1.0,),
        jnp.asarray([0.0, jnp.pi, 3.0 * jnp.pi, 8.0 * jnp.pi]),
    )
    transformed = plan.transform(field)
    result = plan.auto_power(transformed)
    assert bool(result.successful)
    np.testing.assert_allclose(result.shell_values[1], 0.25, rtol=1e-12)
    np.testing.assert_allclose(result.weighted_mode_count[1], 2.0)
    assert int(jnp.sum(result.weighted_mode_count)) == count - 1
    np.testing.assert_allclose(
        jnp.sum(result.shell_values * result.weighted_mode_count),
        result.total_weighted_value,
        rtol=1e-12,
        atol=1e-12,
    )


def test_rfft_parseval_cross_power_and_phase_discrepancy():
    shape = (8, 6)
    x, y = jnp.meshgrid(
        (jnp.arange(shape[0]) + 0.5) / shape[0],
        (jnp.arange(shape[1]) + 0.5) * 2.0 / shape[1],
        indexing="ij",
    )
    field = jnp.cos(2.0 * jnp.pi * x) + 0.25 * jnp.cos(jnp.pi * y)
    shifted = jnp.roll(field, 1, axis=0)
    maximum = jnp.sqrt((jnp.pi * shape[0]) ** 2 + (jnp.pi * shape[1] / 2.0) ** 2)
    plan = phx.discretization.PeriodicFourierShellPlan(
        shape, (1.0, 2.0), jnp.linspace(0.0, maximum, 12)
    )
    first = plan.transform(field)
    second = plan.transform(2.0 * field)
    cross = plan.cross_power(first, second)
    auto = plan.auto_power(first)
    np.testing.assert_allclose(
        cross.shell_values[cross.valid_shells],
        2.0 * auto.shell_values[auto.valid_shells],
        rtol=1e-12,
        atol=1e-12,
    )
    discrepancy = plan.discrepancy(first, plan.transform(shifted))
    real_space = plan.cell_volume * jnp.sum((field - shifted) ** 2)
    np.testing.assert_allclose(discrepancy.total_weighted_value, real_space, rtol=1e-12)
    assert discrepancy.total_weighted_value > 0.0
    np.testing.assert_allclose(
        plan.auto_power(plan.transform(shifted)).shell_values,
        auto.shell_values,
        rtol=1e-12,
        atol=1e-12,
    )


def test_dc_nyquist_final_edge_and_empty_shell_policies():
    plan = phx.discretization.PeriodicFourierShellPlan(
        (4,),
        (1.0,),
        [0.0, 1.0, 2.0 * np.pi, 4.0 * np.pi],
        final_edge_policy="include",
    )
    assert plan.excluded_mode_count == 1
    assert plan.weighted_mode_count[-1] == 3.0
    assert jnp.any(~plan.valid_shells)
    excluded = phx.discretization.PeriodicFourierShellPlan(
        (4,),
        (1.0,),
        [0.0, 2.0 * np.pi, 4.0 * np.pi],
        nyquist_policy="exclude",
    )
    assert excluded.excluded_mode_count == 2


def test_shell_plan_rejects_invalid_shapes_and_supports_gradients():
    with pytest.raises(ValueError, match="invalid"):
        phx.discretization.PeriodicFourierShellPlan((8,), (1.0,), [0.0, 1.0, 1.0])
    plan = phx.discretization.PeriodicFourierShellPlan(
        (8,), (1.0,), [0.0, np.pi, 3.0 * np.pi, 8.0 * np.pi]
    )

    def objective(amplitude):
        field = amplitude * jnp.cos(2.0 * jnp.pi * (jnp.arange(8) + 0.5) / 8.0)
        return plan.auto_power(plan.transform(field)).shell_values[1]

    derivative = jax.grad(objective)(jnp.asarray(1.0))
    np.testing.assert_allclose(derivative, 0.5, rtol=1e-12)
