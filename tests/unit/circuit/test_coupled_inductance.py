import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _plan():
    return phx.circuit.CoupledInductancePlan(
        np.asarray([[2.0, 0.5], [0.5, 1.0]]),
        np.asarray([[0.1, 0.0], [0.0, 0.2]]),
        ("pf-a", "pf-b"),
    ).prepare()


def test_coupled_inductance_implicit_step_closes_energy_ledger():
    prepared = _plan()
    result = prepared.step_implicit_euler(
        np.asarray([1.0, -0.5]), np.asarray([2.0, 1.0]), 0.1
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.ledger.closure_residual_j, 0.0, atol=1.0e-12)
    assert result.energy.stored_energy_j > 0.0
    assert result.energy.resistive_power_w >= 0.0


def test_coupled_inductance_rejects_nonreciprocal_matrix():
    with pytest.raises(ValueError, match="reciprocal"):
        phx.circuit.CoupledInductancePlan(
            np.asarray([[1.0, 0.2], [0.1, 1.0]]),
            np.eye(2),
            ("a", "b"),
        )


def test_coupled_inductance_step_is_differentiable():
    prepared = _plan()

    def final_current(voltage):
        result = prepared.step_implicit_euler(
            jnp.asarray([0.0, 0.0]),
            jnp.asarray([voltage, 0.0]),
            0.1,
        )
        return result.accepted_current_a[0]

    derivative = jax.grad(final_current)(jnp.asarray(1.0))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0
